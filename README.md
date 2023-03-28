# MEC2403_Activities

Repositório para armazemanento de código referente aos trabalhos e listas da disciplina MEC2403 da pós-graduação da PUC-Rio

Aluno: Pedro Henrique Cardoso Paulo

Professor: Ivan Menezes

## Configuração

OS: Ubuntu 20-04 (WSL2)

Python: 3.11 (Miniconda)

## Começando a usar o repositório

### Requisitos de softwares

Para replicar os comandos ensinados neste tutorial, é necessário instalar as seguintes ferramentas gratuitas:

- Ter o git instalado. Pode ser adquirido via este <a href="https://git-scm.com/downloads">link</a>
    - NOTA: para usuários do WSL, o git já vem instalado no subsistema
- Ter o Anaconda 3 ou Miniconda instalado. Pode ser adquirido no site do <a href="https://docs.conda.io/en/latest/miniconda.html">Anaconda.org</a>
- Caso se deseje compilar os relatórios do LaTeX, é necessário instalar alguma distribuição.

Recomendo também a instalação do <a href="https://code.visualstudio.com/Download">VSCode</a> com as extensóes de suporte ao Python e ao Git para melhorar a usabilidade da ferramenta e a integração com o repositório.

### Clonando o repositório

Tendo o git instalado, o conteúdo do repositório pode ser baixado para a sua máquina via o seguinte comando

```(bash)
git clone https://github.com/prj-phcp/MEC2403_Activities.git
```

Atualizações podem ser puxadas do repositório clonado a qualquer momento por meio do comando pull

```(bash)
git pull origin master
```

Caso haja necessidade de atualização do repo, como este é público, basta usar o comando de push

```(bash)
git push origin master
```

Entretanto, para isso é necessário configurar o usuário e e-mail de uma conta git válida

```(bash)
git config user.name "<SEU_NOME_USUARIO>"

git config user.email "<SEU_EMAIL>"
```

- NOTA: A finalidade desse repositório é ser usado para controle de versão de códigos usados em atividades de aula. Por favor, pense bem antes de comitar e, na dúvida, entre em contato antes

### Criando um ambiente

Ter as versões corretas de pacotes é fundamental para garantir a repetibilidade dos exercícios. De modo a permitir isso, um arquivo .yml, contendo as informações dos pacotes utilizados nesses notebooks é encontrado na raiz desse repositório. Para instalá-lo, basta ter o Anaconda ou Miniconda instalado na máquina e rodar a seguinte linha de comando:

```(bash)
conda env create -f environment.yml
```

Para atualizar o arquivo com o ambiente após a instalação de algum pacote novo, executar a seguinte linha com o ambiente ativado:

```(bash)
conda env export | grep -v "^prefix: " > environment.yml
```

## Atividades

As atividades presentes neste repositório são:

- <a href="./Lista0">Lista 00</a>: Exercícios iniciais de programação matemática e quebra de gelo com o Python
- <a href="./Lista1">Lista 01</a>: Implementação inicial dos métodos de busca linear
