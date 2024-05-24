# RAG Travel Recommendation System

This repository contains a Retrieval-Augmented Generation (RAG) based Travel Recommendation System. The system utilizes Qdrant, Sentence Transformers, and a local Large Language Model (LLM) to provide travel recommendations based on hotel reviews.

## Overview

The Travel Recommendation System processes hotel reviews to suggest the best hotels based on user queries. It uses the following technologies:

- **Qdrant**: Vector database for storing and retrieving embeddings.
- **Sentence Transformers**: Model for generating dense vector representations of text.
- **OpenAI API**: Interface for connecting to the local LLM.

## Project Background

This project is based on a course project from Coursera titled "Introduction to Retrieval Augmented Generation (RAG)" by Alfredo Deza. The original code and materials for the course can be found in the [GitHub repository](https://github.com/alfredodeza/learn-retrieval-augmented-generation/tree/main).

## Using Llamafile for Local LLM

The project runs a local LLM using Llamafile. Detailed instructions on how to use Llamafile can be found in the [Mozilla-Ocho Llamafile repository](https://github.com/Mozilla-Ocho/llamafile).

## Dataset

The dataset used for this project is the Hotel Reviews dataset, which contains 515k hotel reviews from Europe. The dataset can be found on Kaggle: [515k Hotel Reviews Data in Europe](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe).
