# Vision Transformer (ViT) en CUDA

Un proyecto académico de alto rendimiento que implementa un **Vision Transformer Híbrido** desde cero en **C++ puro y CUDA**. El modelo está diseñado para clasificar los dígitos de la base de datos MNIST sin depender de ningún framework de Deep Learning (como PyTorch o TensorFlow), gestionando los kernels de GPU, la memoria VRAM y el algoritmo de Backpropagation de forma manual y altamente paralela.

---

## Arquitectura del Modelo

El modelo combina la capacidad de extracción de características locales de las Redes Convolucionales (CNN) con el entendimiento global y secuencial de los Transformers.

1. **CNN Front-End**: Extrae características espaciales de la imagen original. Pasa la imagen 28x28 por una capa convolucional, activación ReLU y Max Pooling.
2. **Patch Embedding**: La imagen transformada es recortada en parches (patches) planos, los cuales se linealizan y se mezclan con su "Positional Encoding" para que el Transformer sepa el orden original.
3. **Transformer Encoder Blocks**: Varios bloques apilados (cada uno con atención paralela `Multi-Head Attention` y una red `MLP` con activación `GELU`) enriquecen la información de cada parche al interactuar entre sí. Se usa `LayerNorm` intensivamente para mantener la estabilidad del gradiente.
4. **Classification Head (Token CLS)**: De toda la secuencia resultante, se extrae únicamente el primer vector (token `[CLS]`) que agrupa el conocimiento de toda la imagen, y se pasa por una capa densa que reduce las dimensiones a las 10 categorías finales mediante la función Softmax.

---

## Organización del Código y Archivos

El repositorio está estructurado en dos capas: `include/` contiene las cabeceras `.cuh` con las declaraciones públicas de cada clase, y `src/` contiene las implementaciones `.cu` con los kernels de CUDA y la lógica completa de cada módulo. A continuación se describe el rol técnico de cada archivo y los pasos que sigue internamente.

---

### Cimientos: Memoria en la GPU

#### `src/Matrix.cu` y `src/Tensor3D.cu`
Son los bloques de construcción fundamentales de todo el sistema. Antes de que cualquier kernel pueda ejecutarse, el dato debe existir en la VRAM de la GPU.

- **`Matrix`**: Representa una matriz 2D (`rows x cols`) cuya memoria vive directamente en la VRAM (`d_data`). Implementa las operaciones algebraicas clave como multiplicación de matrices (`.dot()`), transpuesta, suma, escalado y extracción de filas/columnas (`slice_cols`). Internamente, cada operación lanza un kernel paralelo donde cada hilo procesa uno o varios elementos simultáneamente.
- **`Tensor3D`**: Extiende la misma idea a volúmenes 3D (`channels x height x width`), el formato natural de una imagen. Permite aplanar el volumen a una `Matrix` mediante `.flatten()` y reconstruirlo con `Tensor3D::reconstructureFlatMatrix()`, operaciones esenciales para conectar la CNN con el Transformer.
- Ambas clases implementan **move semantics** de C++17, garantizando que los bloques de VRAM se transfieran sin copias innecesarias entre capas.

---

### Módulo Convolucional: Extracción Visual

#### `src/ConvolutionalLayer.cu`
Implementa la convolución 2D paralela en CUDA. Su flujo interno es:
1. Recibe un `Tensor3D` de entrada (imagen o mapa de características).
2. Lanza un kernel donde cada hilo es responsable de calcular **un único valor de salida**: toma la ventana de `kernel_size x kernel_size` píxeles correspondiente, la multiplica elemento a elemento con los pesos del filtro y suma el resultado.
3. Durante `backward`, calcula el gradiente respecto a los pesos del filtro (`grad_W`) y el gradiente respecto a la entrada (`grad_input`) usando convolución transpuesta, actualizando los filtros con SGD.

#### `src/ActivationLayer.cu` y `src/PoolingLayer.cu`
- **`ActivationLayer` (ReLU)**: Cada hilo aplica `max(0, x)` a un elemento. En `backward` pasa el gradiente solo donde la activación fue positiva (máscara binaria).
- **`PoolingLayer` (Max Pooling)**: Divide el mapa de características en ventanas de `pool_size x pool_size` y cada hilo encuentra el máximo local. Guarda en caché las posiciones de los máximos para el `backward`, donde el gradiente se redirige exclusivamente al elemento que ganó.

#### `src/PatchEmbedding.cu`
Es el puente entre el mundo convolucional y el mundo secuencial del Transformer. Sus pasos internos son:
1. Pasa la imagen por la CNN completa: `ConvolutionalLayer → ReLU → MaxPooling`.
2. **`extract_patches_kernel`**: Divide el mapa de características resultante en una cuadrícula de parches de `patch_h x patch_w`. Cada parche se aplana y se copia como una fila de la matriz de salida.
3. Proyecta linealmente cada parche desde `patch_dim` a `d_model` multiplicando por la matriz `W_proj`.
4. Antepone el token `[CLS]` como la fila 0 de la secuencia, que actuará como el "resumen" global de la imagen.
5. Suma los `pos_embeddings` a toda la secuencia para que el Transformer tenga noción de la posición espacial de cada parche.
6. En `backward`, propaga los gradientes en orden inverso: actualiza los embeddings posicionales, el token CLS, la proyección `W_proj`, reconstruye el gradiente del mapa de características con **`reconstruct_patches_kernel`** (usando `atomicAdd` para evitar condiciones de carrera) y finalmente retropropaga por la CNN.

---

### Módulo Transformer: Razonamiento Global

#### `src/MultiHeadAttention.cu`
Es el componente más complejo matemáticamente. Su ejecución sigue estos pasos:
1. **Proyección QKV**: Multiplica la entrada `X` por tres matrices de pesos (`Wq`, `Wk`, `Wv`) para obtener las matrices Query, Key y Value.
2. **División en cabezas**: Las matrices se dividen en `num_heads` secciones de dimensión `d_k = d_model / num_heads`. Esto permite que cada cabeza aprenda relaciones distintas en paralelo.
3. **Scaled Dot-Product Attention** (por cabeza): Calcula `scores = Q_h · K_h^T / √d_k`. El escalado por `√d_k` previene que los productos internos sean demasiado grandes y saturen el Softmax. Un kernel dedicado aplica el Softmax estable numéricamente (restando el máximo por fila antes de exponenciar). Finalmente, `context = softmax(scores) · V_h`.
4. **Concatenación y proyección de salida**: Los contextos de todas las cabezas se concatenan horizontalmente y se proyectan con `Wo` para retornar a la dimensión `d_model`.

#### `src/LayerNorm.cu`
Normaliza cada vector de la secuencia a media 0 y varianza 1, luego aplica los parámetros aprendibles `γ` (gamma) y `β` (beta). Sus kernels siguen cuatro pasos:
1. **`layernorm_mean_kernel`**: Un bloque por fila, usa shared memory y reducción paralela para calcular la media de cada vector en `O(log D)`.
2. **`layernorm_variance_kernel`**: Mismo patrón, calcula la varianza usando la media ya disponible.
3. **`layernorm_normalize_kernel`**: Normaliza elemento a elemento y aplica `γ` y `β`. Guarda el valor normalizado `x̂` en caché para el backward.
4. **`layernorm_backward_kernel`**: Implementa la fórmula exacta de la derivada de LayerNorm, que requiere dos pasadas de reducción: una para `∑(dx̂)` y otra para `∑(dx̂ · x̂)`. Acumula `grad_gamma` y `grad_beta` con `atomicAdd` y actualiza ambos parámetros con SGD.

#### `src/MultiLayerPerceptron.cu` y `src/TransformerEncoder.cu`
- **`MultiLayerPerceptron`**: Red de dos capas lineales (`d_model → mlp_dim → d_model`) con activación GELU entre ellas. GELU es diferenciable y empíricamente supera a ReLU en Transformers.
- **`TransformerEncoderBlock`**: Ensambla el bloque completo siguiendo la arquitectura Pre-Norm:
  1. `x → LayerNorm1 → MultiHeadAttention → + x` (conexión residual)
  2. `z → LayerNorm2 → MLP → + z` (segunda conexión residual)
  Las conexiones residuales son críticas: permiten que el gradiente fluya directamente hacia capas anteriores sin desvanecerse, haciendo posible el entrenamiento de redes profundas.

---

### Orquestación del Sistema

#### `src/VisionTransformer.cu`
La clase maestra que integra todo el pipeline de extremo a extremo:
- **`forward(imagen)`**: Ejecuta secuencialmente `PatchEmbedding → N×TransformerEncoderBlock → extrae fila [CLS] → LayerNorm final → W_head`. Guarda en caché los resultados intermedios de cada etapa.
- **`backward(y_true)`**: Calcula el gradiente de Cross-Entropy respecto a los logits, retropropaga por la cabeza de clasificación, el LayerNorm final, reconstruye el tensor de gradiente completo colocando el gradiente del CLS en la fila 0 (con ceros en el resto), recorre los bloques encoder en **orden inverso** y finalmente llama al `backward` del PatchEmbedding.
- **`train(...)`**: Itera por épocas, hace shuffle del dataset, procesa muestra a muestra en batches, acumula la pérdida y el accuracy, evalúa en validación y soporta *early stopping* si la pérdida de validación no mejora en N épocas consecutivas.

#### `src/DataLoader.cu`
Responsable de cargar y preparar los datos desde el disco:
- Lee el archivo `.csv` de MNIST línea a línea, parsea los valores separados por coma, normaliza los píxeles al rango `[0, 1]` dividiendo entre 255.
- Convierte cada fila a un `Tensor3D(1, 28, 28)` llamando a `cudaMemcpy` para subir los píxeles directamente a la VRAM.
- Codifica las etiquetas en formato **One-Hot** como una `Matrix` en GPU, el formato que requiere la función de pérdida Cross-Entropy.
- Implementa `accuracy()` que compara el argmax de los logits predichos contra el argmax del label real.

#### `main.cu`
El punto de entrada del programa. Define la arquitectura completa y orquesta el ciclo de vida:
1. Carga los 70,000 registros del MNIST (train + test CSV) y los unifica.
2. Aplica shuffle aleatorio con semilla fija `42` para reproducibilidad.
3. Divide en **70% Train / 15% Validación / 15% Test**.
4. Instancia el `VisionTransformer` con todos sus hiperparámetros.
5. Lanza el entrenamiento e imprime la pérdida y el accuracy por época.
6. Evalúa la precisión final en el conjunto de Test (nunca visto durante el entrenamiento).
7. Genera y guarda `resultados_prediccion.png` usando OpenCV: una cuadrícula 5x5 con 25 imágenes del Test mostrando la predicción del modelo vs. la etiqueta real.

---

## Requisitos y Compilación

### Dependencias
* **NVIDIA CUDA Toolkit**: Versión 11 o superior.
* **Compilador C++**: Compatible con C++17.
* **OpenCV 4**: Para la renderización de la cuadrícula gráfica de predicciones (enlazado mediante `pkg-config`).

### Ejecución
El proyecto está optimizado para compilar rápidamente usando `make`. Por defecto está configurado para tarjetas Ada Lovelace (ej. RTX 4050 con `-arch=sm_89`), pero puede ajustarse en el `Makefile` si tienes otra tarjeta.

```bash
# Limpiar binarios antiguos (opcional)
make clean

# Compilar y ejecutar inmediatamente el entrenamiento
make run
```
Tras el entrenamiento masivo de 5 épocas sobre los datos particionados (Train, Val, Test), el sistema automáticamente generará y guardará el archivo visual **`resultados_prediccion.png`** en la carpeta principal.

![Resultados de Predicción MNIST](resultados_prediccion.png)

---

## El Equipo (Autores y Contribuciones)

Este proyecto fue posible gracias al esfuerzo colectivo del equipo dividiendo el pipeline en 4 módulos integrados milimétricamente en C++:

* **Josue Samuel Philco Puma** *(El Integrador)*
  * Orquestación de la clase maestra `VisionTransformer`, desarrollo de la función de pérdida matemática multiclase (Cross-Entropy).
  * Arquitectura del ciclo de entrenamiento (`train`, `accuracy`, `predict`).
  * `DataLoader` y particionamiento del 70/15/15 del dataset masivo de 70,000 archivos, diseño del Makefile final y generación visual utilizando OpenCV nativo.

* **Johan Fabricio Lizarve Mamani** *(Feature Extractor)*
  * Desarrollo desde cero de la Red Neuronal Convolucional paralela (`ConvolutionalLayer`, `PoolingLayer`, `ActivationLayer`).
  * Lógica del recorte y la estructuración de la imagen en secuencias (`PatchEmbedding`) junto con su *Positional Encoding*.

* **Marko Julio Sumire Ramos** *(Atención Matemática y Memoria)*
  * Construcción de los cimientos en la VRAM de NVIDIA (`Matrix`, `Tensor3D`).
  * Implementación de los kernels de multiplicación de matrices complejas y el módulo core del modelo: El paralelismo atencional de `MultiHeadAttention` y *Scaled Dot-Product Attention*.

* **Erik Manuel Ramos Quispe** *(Encoder Block & Normalización)*
  * Ensamblaje del bloque secuencial `TransformerEncoderBlock` y el flujo de los tensores.
  * Desarrollo del `MultiLayerPerceptron`, cálculo masivo de medias y varianzas en `LayerNorm` y programación de las derivadas complejas de la función de activación `GELU`.
