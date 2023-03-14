# MEC2403_Activities

Repositório para armazemanento de código referente aos trabalhos e listas da disciplina MEC2403 da pós-graduação da PUC-Rio

Aluno: Pedro Henrique Cardoso Paulo
Professor: Ivan

## Criando um ambiente

Ter as versões corretas de pacotes é fundamental para garantir a repetibilidade dos exercícios. De modo a permitir isso, um arquivo .yml, contendo as informações dos pacotes utilizados nesses notebooks é encontrado na raiz desse repositório. Para instalá-lo, basta ter o Anaconda ou Miniconda instalado na máquina e rodar a seguinte linha de comando:

```(bash)
conda env create -f environment.yml
```

Para atualizar o arquivo com o ambiente após a instalação de algum pacote novo, executar a seguinte linha com o ambiente ativado:

```(bash)
conda env export | grep -v "^prefix: " > environment.yml
```
