# malexnet-parking-reproduction
Este projeto tem como objetivo principal reproduzir os resultados do artigo científico *"Deep Learning for Decentralized Parking Lot Occupancy Detection"*.

A replicação deste experimento estruturado serve como um estudo prático de Visão Computacional focado em um problema do mundo real: classificar de forma robusta se uma vaga de estacionamento está ocupada ou vazia, construindo um modelo generalista que funcione para diferentes localidades.

## Organização do Repositório

O fluxo de trabalho deste projeto priorizou a experimentação contínua seguida de refatoração, dividindo o código da seguinte maneira:

*   **`notebooks/`**: Contém os ambientes de exploração e prototipagem, construídos para testar e compreender o funcionamento do PyTorch, das CNNs e do processamento dos dados. **Nota:** *Estes arquivos **não** são necessários para rodar o modelo final.* Eles foram mantidos no repositório por razões exclusivamente acadêmicas e para consultas e estudos futuros.
*   **`src/`**: Contém o código final consolidado. Aqui estão os módulos em Python (`.py`) limpos e otimizados, que compõem a infraestrutura real do modelo. A lógica testada, validada e compreendida nos notebooks foi adaptada e estruturada de forma definitiva nesse diretório.

## Dataset

Para treinar e avaliar o modelo, foi utilizado o dataset **CNRPark-EXT**. Ele é um conjunto de dados de uso comum para a tarefa de detecção de ocupação de vagas de estacionamento utilizando câmeras inteligentes.
O dataset é composto por milhares de recortes de imagens focados em vagas individuais. Essas imagens foram capturadas sob diversas condições climáticas (sol, chuva, neve), diferentes iluminações (dia e noite) e variados ângulos de câmera, o que garante a robustez do modelo em cenários reais.

A classificação das vagas é estritamente binária:
* **0:** Vaga vazia (Empty)
* **1:** Vaga ocupada (Occupied)

## Estrutura de Diretórios

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
## Decisões

### Arquitetura do Modelo (LRN)
Foi notada uma divergência entre as fontes de pesquisa para a reprodução dos resultados. Nesse repositório, uso a implementação do mAlexNet original do artigo, com as operações: **Conv → ReLU → LRN → MaxPool**. Porém, no repositório de reprodução citado, não é utilizado o processo LRN (o que gerou resultados ligeiramente diferentes).

Parâmetros exatos da LRN (`size=5`, `alpha=0.0001`, `beta=0.75`, `k=1.0`) foram extraídos da implementação do autor em seu repositório original.

### Hiperparâmetros de Treino
Seguindo a metodologia do artigo original, usou-se diferentes valores como *learning rate* conforme o treinamento das 18 épocas:
* **0.01**: entre as épocas 1 e 5
* **0.005**: entre as épocas 6 e 11
* **0.0025**: entre as épocas 12 e 18

Usou-se o `weight_decay=0.0005`, que foi um valor encontrado em um artigo mais antigo também de treinamento e que funcionou bem para a reprodução dos resultados. 

Otimizador **SGD** com `momentum=0.9`. Tomou-se a decisão de reaproveitar o mesmo objeto ao longo de todo o treino (ajustando o `lr` manualmente).

### Ambiente e Infraestrutura
O treino foi todo realizado em uma CPU **Ryzen 7 7735HS**, afinal a minha GPU não suporta CUDA.

O uso de `num_workers=8` foi escolhido por meio de um teste empírico de diferentes valores, pois esse foi o que apresentou o processamento mais veloz.

## Como usar o código
### 1. Preparação do Ambiente
Certifique-se de ter o Python instalado. É altamente recomendado o uso de um ambiente virtual (como `venv` ou `conda`) para evitar conflitos de versão. 

Para instalar todas as dependências necessárias para a execução do projeto, rode o comando abaixo na raiz do repositório:

```bash
pip install -r requirements.txt
```
### 3. Execução do Projeto
Com as dependências instaladas e os dados no local correto (checar seção de estrutura de diretórios), você pode executar o treinamento e os testes utilizando os scripts finais localizados na pasta `src/`:

Treinamento: Para iniciar o loop de treino do modelo do zero, execute o script principal a partir da raiz do projeto:

```bash
python src/main.py
```
Teste e Avaliação: Para testar o modelo e extrair as métricas finais de performance, execute:

```bash
python src/testing.py
```

## Comparação de Resultados

**Resultados no conjunto de teste (Nossa Reprodução):**
* **Accuracy:** 98.46%
* **AUC-ROC:** 0.9982

**Comparação com o baseline original:**
* Na publicação de referência *"Deep Learning for Decentralized Parking Lot Occupancy Detection"* (Amato et al.), o modelo mAlexNet treinado e avaliado no dataset CNRPark-EXT atingiu uma acurácia média de 95.70%, com alguns recortes de teste chegando a 97.71%.
* O artigo original afirma que o mAlexNet, mesmo sendo consideravelmente menor, entrega um desempenho comparável ao AlexNet tradicional na tarefa de classificação de vagas. O resultado de **AUC-ROC (0.9982)** confirma essa afirmação, indicando uma ótima capacidade do modelo em distinguir entre vagas livres e ocupadas.

## Referências

### Artigos Científicos
*   **[A Systematic Review on Computer Vision-Based Parking Lot Management Applied on Public Datasets (2022)](https://arxiv.org/abs/2203.06463):** Escrito por Paulo Ricardo Lisboa de Almeida, Jeovane Honório Alves, Rafael Stubs Parpinelli e Jean Paul Barddal.
*   **[PKLot – A robust dataset for parking lot classification (2015)](http://dx.doi.org/10.1016/j.eswa.2015.02.009):** Escrito por Paulo R.L. de Almeida, Luiz S. Oliveira, Alceu S. Britto Jr., Eunelson J. Silva Jr. e Alessandro L. Koerich.
*   **[Car Parking Occupancy Detection Using Smart Camera Networks and Deep Learning (2016)](https://scholar.google.com/scholar?q=Car+Parking+Occupancy+Detection+Using+Smart+Camera+Networks+and+Deep+Learning):** Escrito por Giuseppe Amato, Fabio Carrara, Fabrizio Falchi, Claudio Gennaro, Carlo Meghini e Claudio Vairo.
*   **[Deep Learning for Decentralized Parking Lot Occupancy Detection (2017)](https://scholar.google.com/scholar?q=Deep+Learning+for+Decentralized+Parking+Lot+Occupancy+Detection):** Escrito por Giuseppe Amato, Fabio Carrara, Fabrizio Falchi, Claudio Gennaro, Carlo Meghini e Claudio Vairo.

### Repositórios
*   **[Repositório Oficial do Autor (mAlexNet original em Caffe)](https://github.com/fabiocarrara/deep-parking/tree/master)**
*   **[Repositório Não-Oficial de Reprodução (PyTorch)](https://github.com/wuyenlin/parking_lot_occupancy_detection/tree/master)**
