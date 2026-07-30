# Vision Transformer en CUDA

Este repositorio contiene la implementacion de un Vision Transformer (ViT) hibrido acelerado por hardware utilizando CUDA. El proyecto esta siendo construido desde cero para optimizar el procesamiento en GPU.

## Estructura Actual

El proyecto actualmente incluye la infraestructura base para el procesamiento de imagenes previo a la capa de atencion (el extractor de caracteristicas) y las redes densas. 

Las siguientes piezas han sido implementadas y migradas completamente a CUDA:

1. Modulo CNN (Red Convolucional Base):
   - Tensor3D: Estructura de datos tridimensional que administra su memoria nativamente en VRAM de forma contigua.
   - ConvolutionalLayer: Capa convolucional paralelizada que calcula mapas de caracteristicas fusionando el padding de manera implicita.
   - PoolingLayer: Capa de submuestreo paralelizada con soporte para Max, Min y Average Pooling y propagacion de gradientes mediante operaciones atomicas.
   - ActivationLayer: Capa de activacion ReLU.
   - ConvolutionalNetwork: Envoltorio que ensambla las capas de la red y administra la propagacion.

2. Modulo MLP (Perceptron Multicapa):
   - Matrix: Estructura de matrices bidimensional en CUDA.
   - MultiLayerPerceptron: Red neuronal densa que servira como modulo Feed-Forward dentro del bloque Encoder del Transformer y como clasificador final.

## Proximos Pasos

El desarrollo se centrara a continuacion en el nucleo de la arquitectura Transformer:
- Multi-Head Self-Attention (Mecanismo de Autoatencion).
- Patch Embedding y Positional Encoding.
- Layer Normalization.
- Transformer Encoder Block.

## Compilacion y Uso
(Seccion en desarrollo)
