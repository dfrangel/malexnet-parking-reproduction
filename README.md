# malexnet-parking-reproduction
Esse projeto tem como objetivo replicar os resultados obtidos no artigo "Deep Learning for Decentralized Parking Lot Occupancy Detection".
Replicar os resultados de um experimento estruturado serve como projeto de estudo de visão computacional sobre uma questão real, como o problema de se classificar uma vaga de estacionamento entre "ocupada" ou "vazia" através de um modelo que funcione para qualquer localidade.

O repositório é organizado da seguinte maneira:

  - O folder notebooks contém os arquivos contruídos por mim para compreender melhor funcionamentos de Pytorch, CNN, e quaisquer outros tópicos do projeto. *ELES NÃO SÃO NECESSÁRIOS para o funcionamento final do modelo ou para a verificação de resultados, a decisão de manter seus registros é puramente acadêmica e como forma de estudo próprio no futuro.
  - O folder src é o que possui de fato os módulos necessários para o funcionamento do modelo. Geralmente o fluxo de trabalho é criar um notebook, estudar e testar o funcionamento do código e, após isso, limpar e adaptar ele para um módulo final nesse folder.

# Dataset

Para treinar e avaliar o modelo, foi utilizado o dataset **CNRPark-EXT**. Ele é um conjunto de dados de uso comum para a tarefa de detecção de ocupação de vagas de estacionamento utilizando câmeras inteligentes.
O dataset é composto por milhares de recortes de imagens focados em vagas individuais. Essas imagens foram capturadas sob diversas condições climáticas (sol, chuva, neve), diferentes iluminações (dia e noite) e variados ângulos de câmera, o que garante a robustez do modelo em cenários reais.

A classificação das vagas é estritamente binária:
* **0:** Vaga vazia (Empty)
* **1:** Vaga ocupada (Occupied)

### Estrutura de Diretórios

Para que o DataLoader funcione corretamente, o projeto espera que o dataset seja descompactado dentro de uma pasta `data/` na raiz do projeto, seguindo exatamente esta estrutura:

```
├── data/
│   └── CNRPark-EXT/
│       ├── LABELS/                    # Arquivos de texto com as anotações
│       │   ├── all.txt                # Conjunto de todos os paths com labels
│       │   ├── camera1.txt            
│       │   ├── camera2.txt
│       │   ├── ...                    # O dataset conta com 9 câmeras diferentes
│       │   ├── test.txt               # Imagens e labels para teste final
│       │   ├── train.txt              # Imagens e labels para treinamento
│       │   └── val.txt                # Imagens e labels para validação
│       └── PATCHES/                   # Diretório raiz das imagens das vagas
│           ├── OVERCAST/
│           ├── RAINY/
│           └── SUNNY/
```

Para baixar os arquivos originais e ler mais detalhes sobre a criação e metodologia do dataset, visite o site oficial: [http://cnrpark.it](http://cnrpark.it).

# Decisões

# Como usar o código

# Referêcias e Inspirações
