import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu
# Definir layout "wide"

# Define o dispositivo
device = torch.device("cpu")


# Configurar título e layout
st.set_page_config(page_title="Menu Interativo", layout="wide")

# Botão para remover o cache e dados no session_state
############################################################################
# Criar menu interativo horizontal
selected = option_menu(
    menu_title=None,  # Ocultar título do menu
    options=["home", "Informações", "Tabela", "Gráfico", "Imagens","Limpa Dados"],
    icons=["house", "info-circle", "table", "bar-chart", "image", "trash"],
    menu_icon="cast",  # Ícone do menu
    default_index=0,   # "Selecione..." como padrão
    orientation="horizontal",  # Menu horizontal
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin": "0px",
            "color": "black",
            "--hover-color": "#cce7ff",  # Cor ao passar o mouse
            "border-radius": "5px",
        },
        "nav-link-selected": {
            "background-color": "#cce7ff",  # Cor de fundo do item selecionado
            "color": "black",  # Cor do texto do item selecionado
        },
    }
)
############################################################################
   

# Modelo Gerador
class Generator(torch.nn.Module):
    def __init__(self, z_i, g_n):
        super(Generator, self).__init__()
        self.conv1 = torch.nn.ConvTranspose2d(z_i, g_n*4, 3, 1, 0, bias=False)
        self.batchNorm1 = torch.nn.BatchNorm2d(g_n*4)
        self.conv2 = torch.nn.ConvTranspose2d(g_n*4, g_n*2, 3, 2, 0, bias=False)
        self.batchNorm2 = torch.nn.BatchNorm2d(g_n*2)
        self.conv3 = torch.nn.ConvTranspose2d(g_n*2, g_n, 2, 2, 0, bias=False)
        self.batchNorm3 = torch.nn.BatchNorm2d(g_n)
        self.conv4 = torch.nn.ConvTranspose2d(g_n, 1, 2, 2, 0, bias=False)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.batchNorm1(self.conv1(x)), 0.2)
        x = torch.nn.functional.leaky_relu(self.batchNorm2(self.conv2(x)), 0.2)
        x = torch.nn.functional.leaky_relu(self.batchNorm3(self.conv3(x)), 0.2)
        x = torch.tanh(self.conv4(x))
        return x
    

###########################################################################################
########################FUNÇÃO PARA MOSTRA IMAGENS#########################################
def printImages(images):
    
    # Certifique-se de que existem pelo menos 100 imagens
    num_images = min(len(images), 100)

    # Ajustar o tamanho da figura para acomodar a grade de 10x10 com imagens menores
    fig, axes = plt.subplots(10, 10, figsize=(30, 30))
    axes = axes.flatten()

    for i in range(num_images):
        ax = axes[i]
        ax.set_axis_off()  # Remove os eixos

        # Exibir imagem na sua resolução original
        img_array = images[i].view(28, 28).cpu().numpy()
        ax.imshow(img_array, cmap='gray', interpolation='nearest')

    # Remove os subplots extras
    for j in range(num_images, len(axes)):
        axes[j].set_visible(False)

    # Renderizar o gráfico no Streamlit
    st.pyplot(fig)


 #######################################################################################################
 ##############################FUNÇÃO PARA  CARREGAR DADOS###############################################   
# Função para carregar os dados
def carrrega_dados(classe_selecionado, indice_selecionado):

    # Carrega os vetores latentes
    #caminho_arquivo_2 = "./arquivo_projeto/listar_tensores_300_cada_de_0_9.pt"
    #caminho_arquivo_2 = "./arquivo_projeto/listar_vetores_300_cada_de_0_9.pt"
    caminho_arquivo_2 = "./listar_vetores_300_cada_de_0_9.pt"
    resultados = torch.load(caminho_arquivo_2)

    # Filtra os vetores latentes da classe selecionada
    vetores_da_classe = [
        item['vetor_latente'].view(-1).tolist()
        for item in resultados if item['label'] == classe_selecionado
    ]

    vetor_especificado = vetores_da_classe[indice_selecionado]

    # Cria o DataFrame
    colunas = [f'd{i+1}' for i in range(100)]
    tabela = pd.DataFrame(vetores_da_classe, columns=colunas)

    # Normaliza os dados
    #####################################################
    scaling = StandardScaler()
    Scaled_data = scaling.fit_transform(tabela)

    # Aplica o PCA
    principal = PCA(n_components=1)
    Componentes_principais = principal.fit_transform(Scaled_data)

    # Extrai os pesos do primeiro componente principal
    pesos_do_CP1 = abs(principal.components_[0])
    df = pd.DataFrame(pesos_do_CP1, columns=['Pesos do CP1'])
    df['Dimensões'] = colunas

    # Ordena os pesos em ordem decrescente
    df_ordenado = df.sort_values(by='Pesos do CP1', ascending=False)
    _1_maior_dimensao = df_ordenado.iloc[0]['Dimensões']
    _2_maior_dimensao = df_ordenado.iloc[1]['Dimensões']
    _3_maior_dimensao = df_ordenado.iloc[2]['Dimensões']
    _4_maior_dimensao = df_ordenado.iloc[3]['Dimensões']
    _5_maior_dimensao = df_ordenado.iloc[4]['Dimensões']
    _6_maior_dimensao = df_ordenado.iloc[5]['Dimensões']
###################################################################################################

    df2 = pd.DataFrame({'Dimensões': colunas, 'CP1': pesos_do_CP1}).sort_values(by='CP1', ascending=False)
   
    return   df2, tabela, colunas, vetor_especificado, _1_maior_dimensao, _2_maior_dimensao, _3_maior_dimensao, _4_maior_dimensao,_5_maior_dimensao,_6_maior_dimensao 
    
# Parâmetros
z_i = 100
g_n = 64
classe = list(range(10))
indice = list(range(300))

col1, col2 = st.columns(2)

with col1:
    classe_selecionado = st.selectbox("Selecione a Classe:", classe)
with col2:
    indice_selecionado = st.selectbox("Selecione o Índice:", indice)

    
       
    
@st.cache_data
def cached_data(classe_selecionado, indice_selecionado):
    return carrrega_dados(classe_selecionado, indice_selecionado)

# Botão para carregar dados

df2, tabela, colunas, vetor_especificado, _1_maior_dimensao, _2_maior_dimensao, _3_maior_dimensao, _4_maior_dimensao, _5_maior_dimensao, _6_maior_dimensao = cached_data(classe_selecionado, indice_selecionado)
st.session_state['df2'] = df2
st.session_state['tabela'] = tabela
st.session_state['colunas'] = colunas
st.session_state['vetor_especificado'] = vetor_especificado
st.session_state['_1_maior_dimensao'] = _1_maior_dimensao
st.session_state['_2_maior_dimensao'] = _2_maior_dimensao
st.session_state['_3_maior_dimensao'] = _3_maior_dimensao
st.session_state['_4_maior_dimensao'] = _4_maior_dimensao
st.session_state['_5_maior_dimensao'] = _5_maior_dimensao
st.session_state['_6_maior_dimensao'] = _6_maior_dimensao
st.success("Dados carregados e armazenados!")

# Botão para usar dados carregados

df2 = st.session_state['df2']
tabela = st.session_state['tabela']
_1_maior_dimensao = st.session_state['_1_maior_dimensao']
maior_valor1 = round(tabela[_1_maior_dimensao].max())
menor_valor1 = round(tabela[_1_maior_dimensao].min())
##########################################################
_2_maior_dimensao = st.session_state['_2_maior_dimensao']
maior_valor2 = round(tabela[_2_maior_dimensao].max())
menor_valor2 = round(tabela[_2_maior_dimensao].min())
##########################################################
_3_maior_dimensao = st.session_state['_3_maior_dimensao']
maior_valor3 = round(tabela[_3_maior_dimensao].max())
menor_valor3 = round(tabela[_3_maior_dimensao].min())
##########################################################
_4_maior_dimensao = st.session_state['_4_maior_dimensao']
maior_valor4 = round(tabela[_4_maior_dimensao].max())
menor_valor4 = round(tabela[_4_maior_dimensao].min())
##########################################################
_5_maior_dimensao = st.session_state['_5_maior_dimensao']
maior_valor5 = round(tabela[_5_maior_dimensao].max())
menor_valor5 = round(tabela[_5_maior_dimensao].min())
##########################################################
_6_maior_dimensao = st.session_state['_6_maior_dimensao']
maior_valor6 = round(tabela[_6_maior_dimensao].max())
menor_valor6 = round(tabela[_6_maior_dimensao].min())



###################################################################################################

###################################################################################################





# Redirecionar com base na seleção

if selected == "home":
    st.write("")
elif selected == "Informações":
    st.markdown(f"# As maiores Dimensões da classe {classe_selecionado}")
    st.write(f"O Menor e o Maior valor da dimensão {_1_maior_dimensao} é {menor_valor1} e {maior_valor1}")
    st.write(f"O Menor e o Maior valor da dimensão {_2_maior_dimensao} é {menor_valor2} e {maior_valor2}")
    st.write(f"O Menor e o Maior valor da dimensão {_3_maior_dimensao} é {menor_valor3} e {maior_valor3}")
    st.write(f"O Menor e o Maior valor da dimensão {_4_maior_dimensao} é {menor_valor4} e {maior_valor5}")
    st.write(f"O Menor e o Maior valor da dimensão {_5_maior_dimensao} é {menor_valor5} e {maior_valor5}")
    st.write(f"O Menor e o Maior valor da dimensão {_6_maior_dimensao} é {menor_valor6} e {maior_valor6}")
elif selected == "Tabela":
    st.markdown(f"# Ver Tabela da Classe {classe_selecionado}")
    st.dataframe(tabela)  # Tabela interativa com rolagem
elif selected == "Gráfico":
    st.markdown(f"# Ver Grafico da Classe {classe_selecionado}")
    plt.figure(figsize=(30, 15))  # Aumentar bastante a figura
    plt.bar(df2['Dimensões'], df2['CP1'], color="red")

    # Títulos e rótulos em negrito
    plt.xlabel('Dimensões', fontweight='bold', fontsize=12)
    plt.ylabel('Peso das variáveis na CP1', fontweight='bold', fontsize=12)
    plt.title(f'Dimensões da CP1, da classe {classe_selecionado}', fontweight='bold', fontsize=20)

    # Ajustar rótulos e margens
    plt.xticks(rotation=90, ha='right', fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    plt.subplots_adjust(bottom=0.3)

    # Exibir o gráfico no Streamlit
    st.pyplot(plt)

        

elif selected == "Imagens":
    st.markdown(f"# Ver Imagens da Classe {classe_selecionado}")

    
    # Inicialize o modelo
    G = Generator(z_i, g_n).to(device)
    #caminho_arquivo_1 = "./arquivo_projeto/Modelo_salvo_G_30_epochs.pt"
    caminho_arquivo_1 = "./Modelo_salvo_G_30_epochs.pt"
    G.load_state_dict(torch.load(caminho_arquivo_1))
    G.eval()

    ##############################################################

    vetor_original = tabela.loc[indice_selecionado].copy()  # Selecionar o vetor original (antes da alteração)
    trans_vetor_original_array = vetor_original.to_numpy()  # Converter o vetor original para um array NumPy
    formata_vetor_original = trans_vetor_original_array.reshape(1, 100, 1, 1)  # Redimensionar para (1, 100, 1, 1)

    # Adiciona o slider para controlar o valor
    valor_dimensao1 = vetor_original[_1_maior_dimensao]
    valor_dimensao2 = vetor_original[_2_maior_dimensao]
    valor_dimensao3 = vetor_original[_3_maior_dimensao]
    valor_dimensao4 = vetor_original[_4_maior_dimensao]
    valor_dimensao5 = vetor_original[_5_maior_dimensao]
    valor_dimensao6 = vetor_original[_6_maior_dimensao]

    col1, col2, col3 = st.columns(3)
    ########### GERAÇÃO DAS IMAGENS ##############
    with torch.no_grad():
        image_original = G(torch.tensor(formata_vetor_original, dtype=torch.float32))  # Imagem com vetor original
    with col1:
        st.info("Imagem Original")
        printImages([image_original])
    with col2:
        dimensao1 = st.slider(
            f'Dimensão {_1_maior_dimensao}',
            min_value=float(menor_valor1),
            max_value=float(maior_valor1),
            value=float(valor_dimensao1),
            step=0.01
        )
        dimensao2 = st.slider(
            f'Dimensão {_2_maior_dimensao}',
            min_value=float(menor_valor2),
            max_value=float(maior_valor2),
            value=float(valor_dimensao2),
            step=0.01
        )
        dimensao3 = st.slider(
            f'Dimensão {_3_maior_dimensao}',
            min_value=float(menor_valor3),
            max_value=float(maior_valor3),
            value=float(valor_dimensao3),
            step=0.01
        )
        dimensao4 = st.slider(
            f'Dimensão {_4_maior_dimensao}',
            min_value=float(menor_valor4),
            max_value=float(maior_valor4),
            value=float(valor_dimensao4),
            step=0.01
        )
        dimensao5 = st.slider(
            f'Dimensão {_5_maior_dimensao}',
            min_value=float(menor_valor5),
            max_value=float(maior_valor5),
            value=float(valor_dimensao5),
            step=0.01
        )
        dimensao6 = st.slider(
            f'Dimensão {_6_maior_dimensao}',
            min_value=float(menor_valor6),
            max_value=float(maior_valor6),
            value=float(valor_dimensao6),
            step=0.01
        )

        # Atualiza o valor na tabela com o valor do slider
        tabela.loc[indice_selecionado, _1_maior_dimensao] = dimensao1
        tabela.loc[indice_selecionado, _2_maior_dimensao] = dimensao2
        tabela.loc[indice_selecionado, _3_maior_dimensao] = dimensao3
        tabela.loc[indice_selecionado, _4_maior_dimensao] = dimensao4
        tabela.loc[indice_selecionado, _5_maior_dimensao] = dimensao5
        tabela.loc[indice_selecionado, _6_maior_dimensao] = dimensao6

        ########### VETOR ALTERADO ##############
        vetor_alterado = tabela.loc[indice_selecionado]  # Selecionar o vetor alterado (após a alteração)
        trans_vetor_alterado_array = vetor_alterado.to_numpy()  # Converter o vetor alterado para um array NumPy
        formata_vetor_alterado = trans_vetor_alterado_array.reshape(1, 100, 1, 1)  # Redimensionar para (1, 100, 1, 1)

        ########### GERAÇÃO DAS IMAGENS ##############
        with torch.no_grad():
            image_alterada = G(torch.tensor(formata_vetor_alterado, dtype=torch.float32))  # Imagem com vetor alterado

        with col3: 
            st.info("Imagem Alterada")   
            printImages([image_alterada])
            
elif selected == "Limpa Dados":
    st.cache_data.clear()  # Limpa o cache armazenado
    st.session_state.clear()  # Limpa todas as variáveis no session_state
    st.success("Cache e dados removidos com sucesso!")

############################################




