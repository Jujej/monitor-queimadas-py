# 🌍 Pyrosense ML - Sistema de Monitoramento e Análise de Queimadas

Aplicação web desenvolvida em Python (Flask) que permite o upload de bases de dados de focos de incêndio abertas do INPE, oferecendo análises visuais interativas, mapas e predições com Machine Learning.

## 📌 Tema do Projeto
Utilizamos uma base de dados abertos do INPE sobre registros de queimadas detectados por satélite. O objetivo é fornecer insights preditivos e análises geoespaciais e temporais detalhadas sobre os focos de incêndio.

## 🔍 Escopo da Análise de Dados
Na nossa análise, a aplicação apresenta:
- **Previsão do risco de incêndio** no estado.
- **Identificação do município de maior criticidade** e sua variação em relação ao mês anterior.
- **Ranking dos municípios** com maior volume de focos de calor.
- **Heatmap de sazonalidade** correlacionando mês × ano.

## 🚀 Funcionalidades da Aplicação
- **Upload Flexível de Dados:** Suporte a diferentes formatos de arquivos estruturados (.csv, .json, .xml, .kml).
- **Análise de Dados e Visualização:** Geração automática de gráficos estatísticos e visualização espacial dos focos.
- **Machine Learning Dinâmico:** Escolha de classificadores (Árvore de Decisão, KNN, Random Forest) com ajuste de hiperparâmetros diretamente pela interface.
- **Treinamento em Tempo Real:** Capacidade de re-treinar o modelo dinamicamente sempre que novos dados forem inseridos.

---

## 🛠️ Tecnologias Utilizadas
- **Linguagem:** Python 3.x
- **Framework Web:** Flask
- **Manipulação e Análise de Dados:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (com suporte a motor nativo de fallback)
- **Front-end:** HTML5, CSS3, JavaScript (Templates do Flask)

---

## 👥 Equipe e Divisão de Tarefas
Front-end e Layout: Desenvolvimento da interface web, estilização e integração visual.

Análise de Dados e Back-end (Pandas/Flask): Tratamento das bases do INPE, rotas de upload e estruturação das análises.

Machine Learning e Documentação (README): Implementação dos modelos preditivos, classificadores, testes dinâmicos e documentação do sistema.

---

## 📦 Como Instalar e Executar o Projeto

Siga os passos abaixo para configurar e rodar o projeto na sua máquina:

### 1. Clonar o repositório

git clone (https://github.com/Jujej/monitor-queimadas-py.git)
cd monitor-queimadas-py

### 2. Criar e ativar um Ambiente Virtual (Recomendado)

python -m venv venv
# No Windows:
venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate

### 3. Instalar as dependências

pip instald -r requirements.txt

### 4. Executar a aplicação

python app.py