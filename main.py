import streamlit as st


# Funções auxiliares para ajustar os valores
def adjust_value(key, increment=True):
    if increment and st.session_state[key] < 255:
        st.session_state[key] += 1
    elif not increment and st.session_state[key] > 0:
        st.session_state[key] -= 1

# Inicializa os valores dos vetores se não existirem na sessão
if 'vetor_1' not in st.session_state:
    st.session_state['vetor_1'] = 128  # Valor inicial padrão

# Primeiro vetor latente com botões de controle
st.write("#### Variações do primeiro vetor latente")
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if st.button('Menos', key='decrease_v1'):
        adjust_value('vetor_1', increment=False)

with col2:
    vetor_1 = st.slider(
        'Escolha um valor entre 0 e 255',
        min_value=0,
        max_value=255,
        value=st.session_state['vetor_1'],
        key="vetor_1_slider"
    )
    # Atualiza o valor da sessão quando o slider é movido
    st.session_state['vetor_1'] = vetor_1

with col3:
    if st.button('Mais', key='increase_v1'):
        adjust_value('vetor_1', increment=True)


