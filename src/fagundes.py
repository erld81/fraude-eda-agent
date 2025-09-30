import streamlit as st
import pandas as pd
import zipfile
import os
import google.generativeai as genai
import matplotlib.pyplot as plt
import io
import contextlib
import unicodedata
import hashlib
import faiss
import numpy as np
import pickle
import tempfile
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
# Importação da SentenceTransformer será feita via st.cache_resource

# --- Configurações Iniciais e Variáveis ---
# Streamlit Page Configuration
st.set_page_config(
    page_title="Análise Exploratória de Dados (EDA) com Gemini e RAG",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes
CHUNK_SIZE = 1000

# --- Inicialização de Session State ---
if 'gemini_api_key' not in st.session_state:
    st.session_state['gemini_api_key'] = ''
if 'zip_bytes' not in st.session_state:
    st.session_state['zip_bytes'] = None
if 'zip_hash' not in st.session_state:
    st.session_state['zip_hash'] = None
    
# CORREÇÃO DO KeyError: "available_files"
if 'available_files' not in st.session_state:
    st.session_state['available_files'] = [] 
    
if 'file_options_map' not in st.session_state:
    st.session_state['file_options_map'] = {}
if 'selected_file_name' not in st.session_state:
    st.session_state['selected_file_name'] = None
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'df_columns' not in st.session_state:
    st.session_state['df_columns'] = None
if 'faiss_index' not in st.session_state:
    st.session_state['faiss_index'] = None
if 'documents' not in st.session_state:
    st.session_state['documents'] = []
if 'total_lines' not in st.session_state:
    st.session_state['total_lines'] = 0
if 'processed_percentage' not in st.session_state:
    st.session_state['processed_percentage'] = 0
if 'current_chunk_start' not in st.session_state:
    st.session_state['current_chunk_start'] = 0
if 'cleaned_status' not in st.session_state:
    st.session_state['cleaned_status'] = {}
if 'file_name_context' not in st.session_state:
    st.session_state['file_name_context'] = ""
if 'conclusoes_historico' not in st.session_state:
    st.session_state['conclusoes_historico'] = ""
if 'codigo_gerado' not in st.session_state:
    st.session_state['codigo_gerado'] = None
if 'resultado_texto' not in st.session_state:
    st.session_state['resultado_texto'] = None
if 'resultado_df' not in st.session_state:
    st.session_state['resultado_df'] = None
if 'erro_execucao' not in st.session_state:
    st.session_state['erro_execucao'] = None
if 'img_bytes' not in st.session_state:
    st.session_state['img_bytes'] = None
if 'consultar_ia' not in st.session_state:
    st.session_state['consultar_ia'] = False
if 'exibir_codigo' not in st.session_state:
    st.session_state['exibir_codigo'] = False
if 'habilitar_grafico' not in st.session_state:
    st.session_state['habilitar_grafico'] = False
if 'gerar_pdf' not in st.session_state:
    st.session_state['gerar_pdf'] = False
if 'user_query_input_widget' not in st.session_state:
    st.session_state['user_query_input_widget'] = ""
if 'current_query_text' not in st.session_state:
    st.session_state['current_query_text'] = ""


# --- Helper Functions ---
def normalize_text(text):
    """Normaliza texto (remove acentos e cedilhas)."""
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

# --- Agente 0: Clarificação de Consultas (NOVO AGENTE) ---
def agente0_clarifica_pergunta(pergunta_original, api_key):
    """Usa o Gemini para corrigir erros de digitação e clarificar a intenção."""
    if not api_key:
        return pergunta_original

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
# INSTRUÇÕES:
Você é um Clarificador de Consultas. Sua única função é corrigir erros de digitação e tornar a consulta do usuário o mais clara e objetiva possível, SEM alterar o significado original. Sua saída DEVE ser APENAS a consulta corrigida/clarificada.

# EXEMPLOS DE CORREÇÃO:
USUÁRIO: "Qual o tipu de cada colna?"
SAÍDA: "Qual o tipo de cada coluna?"

USUÁRIO: "ttata o arquiu?"
SAÍDA: "Do que se trata o arquivo?"
        
# CONSULTA ORIGINAL DO USUÁRIO:
{pergunta_original}

# CONSULTA CLARIFICADA:
"""
        response = model.generate_content(prompt)
        # Limita para garantir que seja apenas uma frase
        return response.text.strip().split('\n')[0]
    
    except Exception:
        # Em caso de erro, retorna a pergunta original para não bloquear o fluxo
        return pergunta_original

# --- RAG Components ---
@st.cache_resource
def load_embedding_model():
    """Carrega o modelo de embedding uma única vez."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

def create_faiss_index_for_chunk(chunk):
    """Cria/adiciona a um índice FAISS para um chunk específico."""
    model = load_embedding_model()
    
    # 1. Pré-processamento
    docs_chunk = chunk.astype(str).apply(lambda x: ' '.join(x), axis=1).tolist()
    
    # Verifica se há documentos para processar
    if not docs_chunk:
        return True

    # 2. Embedding
    embeddings_chunk = model.encode(docs_chunk, show_progress_bar=False)
    
    dimension = embeddings_chunk.shape[1]
    
    # 3. Criação/Adição ao Índice FAISS
    if st.session_state['faiss_index'] is not None:
        st.session_state['faiss_index'].add(np.array(embeddings_chunk).astype('float32'))
    else:
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings_chunk).astype('float32'))
        st.session_state['faiss_index'] = index
        
    # 4. Atualiza Documentos
    if st.session_state['documents'] is None:
        st.session_state['documents'] = []
    st.session_state['documents'].extend(docs_chunk)
    
    return True

def retrieve_context(query, index, documents, top_k=3):
    """Recupera os documentos mais relevantes do índice FAISS para uma dada consulta."""
    model = load_embedding_model()
    query_embedding = model.encode([query])
    
    if index is None or index.ntotal == 0:
        return ""
    
    # Faiss espera np.float32, então convertemos a query embedding
    D, I = index.search(np.array(query_embedding).astype('float32'), top_k)
    
    retrieved_docs = [documents[i] for i in I[0] if i < len(documents)]
    
    return "\n".join(retrieved_docs)

def save_progress(file_hash, df, faiss_index, documents, total_lines):
    """Salva o progresso no disco."""
    try:
        if st.session_state.get('selected_file_name'):
            unique_file_hash = hashlib.md5((file_hash + st.session_state['selected_file_name']).encode()).hexdigest()
            
            temp_dir = tempfile.gettempdir()
            
            # Garante que o df não está vazio antes de salvar
            if df is not None and not df.empty:
                with open(os.path.join(temp_dir, f"{unique_file_hash}_df.pkl"), "wb") as f:
                    pickle.dump(df, f)
            
            # Garante que o índice foi criado antes de salvar
            if faiss_index is not None and faiss_index.ntotal > 0:
                faiss.write_index(faiss_index, os.path.join(temp_dir, f"{unique_file_hash}_faiss_index.bin"))
                
            if documents:
                with open(os.path.join(temp_dir, f"{unique_file_hash}_documents.pkl"), "wb") as f:
                    pickle.dump(documents, f)
            
            with open(os.path.join(temp_dir, f"{unique_file_hash}_metadata.txt"), "w") as f:
                f.write(str(total_lines))
                
            return True
        return False
    except Exception as e:
        # st.error(f"Erro ao salvar o progresso: {e}") 
        return False

def load_progress(file_hash, selected_file_name):
    """Carrega o progresso do disco, se existir."""
    try:
        unique_file_hash = hashlib.md5((file_hash + selected_file_name).encode()).hexdigest()
        temp_dir = tempfile.gettempdir()
        
        df_path = os.path.join(temp_dir, f"{unique_file_hash}_df.pkl")
        faiss_path = os.path.join(temp_dir, f"{unique_file_hash}_faiss_index.bin")
        docs_path = os.path.join(temp_dir, f"{unique_file_hash}_documents.pkl")
        meta_path = os.path.join(temp_dir, f"{unique_file_hash}_metadata.txt")

        # Verifica se todos os arquivos essenciais existem
        if os.path.exists(df_path) and os.path.exists(faiss_path) and os.path.exists(docs_path):
            with open(df_path, "rb") as f:
                df = pickle.load(f)
            faiss_index = faiss.read_index(faiss_path)
            with open(docs_path, "rb") as f:
                documents = pickle.load(f)
            
            total_lines = 0
            if os.path.exists(meta_path):
                 with open(meta_path, "r") as f:
                    total_lines = int(f.read())
            
            # Retorna o total de linhas do DF carregado, que é o número real de linhas processadas
            return df, faiss_index, documents, len(df)
        return None, None, None, 0
    except Exception as e:
        # print(f"Erro ao carregar o progresso: {e}")
        return None, None, None, 0

# --- Agente 1: Funções de Processamento de Arquivos ---
def agente1_identifica_arquivos(zip_bytes):
    """
    Identifica todos os arquivos CSV, XLSX e TXT no ZIP e tenta obter cabeçalhos.
    """
    files_info = []
    
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        for name in z.namelist():
            if name.startswith('__MACOSX/') or name.endswith('/'):
                continue
                
            ext = os.path.splitext(name)[1].lower()
            
            if ext in ['.csv', '.xlsx', '.txt']:
                try:
                    with z.open(name, 'r') as file_in_zip:
                        data_in_memory = io.BytesIO(file_in_zip.read())
                        
                        header = []
                        if ext == '.csv':
                            temp_df = pd.read_csv(data_in_memory, nrows=0, encoding='utf-8', on_bad_lines='skip', low_memory=False)
                            header = temp_df.columns.tolist()
                        
                        elif ext == '.xlsx':
                            temp_df = pd.read_excel(data_in_memory, nrows=0)
                            header = temp_df.columns.tolist()
                        
                        elif ext == '.txt':
                            data_in_memory.seek(0)
                            # Tentativa de ler a primeira linha para inferir o separador
                            first_line = io.TextIOWrapper(data_in_memory, encoding='utf-8').readline().strip()
                            data_in_memory.seek(0)
                            
                            separator = '\s+'
                            if ',' in first_line:
                                separator = ','
                            elif ';' in first_line:
                                separator = ';'

                            temp_df = pd.read_csv(data_in_memory, nrows=1, encoding='utf-8', sep=separator, engine='python', on_bad_lines='skip', header=None)
                            
                            if temp_df.shape[1] > 0:
                                header = [f"COL_{i+1}" for i in range(temp_df.shape[1])]
                            else:
                                header = ["COL_1"] 
                                
                        file_info = {
                            "name": name,
                            "extension": ext,
                            "header": header,
                            "schema_text": ", ".join(header),
                            "num_cols": len(header)
                        }
                        files_info.append(file_info)
                except Exception as e:
                    pass
            
    return files_info

def agente1_interpreta_contexto_arquivo(api_key, file_info_list):
    """
    Usa o Gemini para descrever o que cada arquivo representa com base no nome e cabeçalho.
    """
    if not api_key:
        return {info["name"]: "API Key não configurada para gerar contexto." for info in file_info_list}

    contextos = {}
    try:
        genai.configure(api_key=api_key)
        # MODELO ATUALIZADO PARA gemini-2.5-flash
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt_parts = ["# PERSONA: Você é um Analista de Dados Sênior. Sua única função é INFERIR o CONTEÚDO e CONTEXTO de um arquivo de dados baseado no NOME e CABEÇALHO. DÊ UMA DESCRIÇÃO DE UMA ÚNICA FRASE CURTA. \n\n# ARQUIVOS PARA ANÁLISE:\n"]
        
        for info in file_info_list:
            prompt_parts.append(f"- ARQUIVO: {info['name']} (Colunas: {info['schema_text']})\n")
        
        prompt_parts.append("\n# INFERÊNCIA:\nResponda APENAS com uma lista numerada, onde cada item é uma descrição concisa (uma frase) para o respectivo arquivo, focando no que ele representa. Ex: 'O arquivo representa dados de transações de cartão de crédito e a coluna CLASS indica fraude.'\n")
        
        response = model.generate_content("".join(prompt_parts))
        
        descricoes = [line.strip() for line in response.text.split('\n') if line.strip().startswith(('1.', '2.', '3.', '-', '*')) or (len(line.strip()) > 5 and i > 0)]
        
        for i, info in enumerate(file_info_list):
            if i < len(descricoes):
                text = descricoes[i]
                if text and text[0].isdigit() and '.' in text[:3]:
                    text = text.split('.', 1)[-1].strip()
                contextos[info["name"]] = text
            else:
                contextos[info["name"]] = f"Erro na inferência automática. Cabeçalho: {info['schema_text']}"
                
    except Exception as e:
        for info in file_info_list:
             contextos[info["name"]] = f"Erro na inferência automática. Cabeçalho: {info['schema_text']}"
            
    return contextos

def agente1_processa_arquivo_chunk(zip_bytes, selected_file_name, start_row, nrows, df_columns, expected_num_cols):
    """
    Processa um chunk do arquivo selecionado (CSV, XLSX, TXT) dentro do ZIP.
    """
    ext = os.path.splitext(selected_file_name)[1].lower()
    
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
            with z.open(selected_file_name, 'r') as file_in_zip:
                file_bytes_in_memory = io.BytesIO(file_in_zip.read())
                
                chunk = pd.DataFrame()
                
                # Configurações de leitura
                skiprows = range(1, start_row + 1) if start_row > 0 else 0
                header = None if start_row > 0 else 'infer'

                # Leitura de CSV
                if ext == '.csv':
                    chunk = pd.read_csv(
                        file_bytes_in_memory,
                        skiprows=skiprows,
                        nrows=nrows,
                        low_memory=False,
                        header=header,
                        encoding='utf-8',
                        on_bad_lines='skip'
                    )
                
                # Leitura de XLSX
                elif ext == '.xlsx':
                    chunk = pd.read_excel(file_bytes_in_memory, header=header, skiprows=skiprows, nrows=nrows)
                        
                # Leitura de TXT
                elif ext == '.txt':
                    # Tenta inferir o separador para leitura do chunk
                    file_bytes_in_memory.seek(0)
                    first_line = io.TextIOWrapper(file_bytes_in_memory, encoding='utf-8', errors='ignore').readline().strip()
                    file_bytes_in_memory.seek(0)
                    
                    separator = '\s+'
                    if ',' in first_line:
                        separator = ','
                    elif ';' in first_line:
                        separator = ';'
                        
                    chunk = pd.read_csv(
                        file_bytes_in_memory,
                        skiprows=skiprows,
                        nrows=nrows,
                        low_memory=False,
                        header=header,
                        encoding='utf-8',
                        sep=separator,
                        engine='python',
                        on_bad_lines='skip'
                    )

                if chunk.empty:
                    return None, "Processamento de todos os lotes concluído."

                # --- TRATAMENTO DE COLUNAS/ESQUEMA ---
                if start_row == 0:
                    # Captura o cabeçalho original (antes da normalização)
                    if df_columns is None:
                        st.session_state['df_columns'] = chunk.columns
                    expected_num_cols = len(st.session_state['df_columns'])
                    
                
                if df_columns is not None:
                    # Correção de Length Mismatch para chunks subsequentes
                    current_cols = chunk.shape[1]
                    
                    if current_cols < expected_num_cols:
                        for i in range(current_cols, expected_num_cols):
                            chunk[f'TEMP_FILL_{i}'] = pd.NA
                        chunk = chunk.iloc[:, :expected_num_cols]
                    
                    elif current_cols > expected_num_cols:
                        chunk = chunk.iloc[:, :expected_num_cols]
                        
                    # Atribui os nomes de coluna originais (normalizados)
                    chunk.columns = [normalize_text(col.strip().upper()) for col in st.session_state['df_columns']]
                else:
                    # Normalização das colunas
                    chunk.columns = [normalize_text(col.strip().upper()) for col in chunk.columns]

                return chunk, "Dados carregados e prontos para análise!"
            
    except Exception as e:
        return None, f"Erro ao processar o arquivo: header must be integer or list of integers {e}"

# --- Agente de Limpeza de Dados ---
def agente_limpeza_dados(df):
    """
    Identifica e converte colunas para tipos numéricos e categóricos.
    Aplica a limpeza 'in-place' no DF.
    """
    if df is None:
        return None

    # Iterar sobre uma cópia da lista de colunas para evitar problemas de modificação durante o loop
    for col in list(df.columns):
        if col not in df.columns: # Proteção caso a coluna seja excluída ou renomeada
             continue

        temp_series = pd.to_numeric(df[col], errors='coerce')
        
        # 1. Numérico
        if temp_series.notna().sum() / len(temp_series) > 0.8:
            df[col] = temp_series
            st.session_state['cleaned_status'][col] = 'Numeric'
        
        # 2. Categórico
        elif df[col].nunique() < 50 and len(df[col].unique()) < len(df) / 2:
            df[col] = df[col].astype('category')
            st.session_state['cleaned_status'][col] = 'Categorical'
        
        # 3. Texto/Objeto
        else:
            st.session_state['cleaned_status'][col] = 'Object'
    
    return df

# --- Agente 2 & 4: Geração de Código e Conclusão ---
def agente2_gera_codigo_pandas_eda(pergunta, api_key, df, retrieved_context=None, historico_conclusoes=None, file_context=None):
    """Gera código Pandas para EDA e a conclusão em linguagem natural."""
    if df is None:
        return "Erro: DataFrame não carregado. Faça o upload do arquivo primeiro.", None

    if not api_key:
        return "Erro: Chave da API do Gemini não fornecida.", None

    try:
        genai.configure(api_key=api_key)
        # MODELO ATUALIZADO PARA gemini-2.5-flash
        model = genai.GenerativeModel('gemini-2.5-flash')

        schema = '\n'.join([f"- {c} (dtype: {df[c].dtype})" for c in df.columns])

        pergunta_limpa = normalize_text(pergunta).upper()
        
        # 1. PERGUNTAS SOBRE TIPOS/ESQUEMA DE COLUNAS (Gera um DataFrame tabular VERTICAL)
        if any(keyword in pergunta_limpa for keyword in ["QUE TIPO DE DADOS", "QUAIS OS TIPOS DE COLUNAS", "DTYPE COLUNAS", "TIPOS DE DADOS NAS COLUNAS", "TIPOS DAS COLUNAS"]):
            
            codigo_gerado = f"""
# Cria um DataFrame vertical com duas colunas
schema_df = pd.DataFrame(df.dtypes).reset_index()
schema_df.columns = ['NOME_DA_COLUNA', 'TIPO_DE_DADO']

# Atribui para visualização tabular no Streamlit
resultado_df = schema_df
print(resultado_df.to_string(index=False)) # Imprime sem o índice para limpeza
"""
            conclusoes = "A análise revela o tipo de dado de cada coluna no conjunto de dados, auxiliando na verificação de consistência e na preparação para modelagem."
            return codigo_gerado, conclusoes
            
        # 2. PERGUNTAS SOBRE CONTEXTO GERAL (Gera apenas texto que será convertido em tabela 1x1)
        if any(keyword in pergunta_limpa for keyword in ["QUE SE TRATA O ARQUIVO", "CONTEUDO DO ARQUIVO", "REPRESENTA O ARQUIVO", "O QUE E ESSE DATASET"]):
            prompt_interpretacao = f"""
# PERSONA E OBJETIVO PRINCIPAL
Você é um Analista de Dados Sênior. Sua única função é INTERPRETAR e DESCREVER o conteúdo de um DataFrame. Sua saída DEVE ser APENAS uma descrição textual detalhada (qualitativa), sem código ou comentários.

# CONTEXTO DO DATAFRAME `df`
Esquema do DataFrame:
{schema}
CONTEXTO DO NOME DO ARQUIVO: '{file_context}'.

# PERGUNTA DO USUÁRIO
{pergunta}

# DESCRIÇÃO FINAL DO CONTEÚDO
"""
            response = model.generate_content(prompt_interpretacao)
            texto_limpo = response.text.replace("'", "\\'").replace('"', '\\"').replace('\n', ' ').strip()
            # O executor de código irá criar um resultado_df a partir deste print para garantir a tabela.
            codigo_gerado = f"print('{texto_limpo}')"
            
            # Conclusão customizada para contextualização (sem contagem errada de linhas)
            conclusoes_contexto = f"O arquivo '{file_context}' foi contextualizado. Ele contém {len(df)} registros."
            
            return codigo_gerado, conclusoes_contexto
            
        # 3. CORREÇÃO ESPECÍFICA PARA BOXPLOT e HISTOGRAMA (Evita múltiplas figuras e layout fixo)
        if any(keyword in pergunta_limpa for keyword in ["OUTLIER", "BOXPLOT", "DISPERSAO", "HISTOGRAMA", "DISTRIBUICAO"]):
             
            plot_type = 'boxplot' if any(k in pergunta_limpa for k in ["OUTLIER", "BOXPLOT"]) else 'hist'
            plot_func = 'df.boxplot(column=col, ax=axes[i], grid=False)' if plot_type == 'boxplot' else 'axes[i].hist(df[col].dropna(), bins=20, edgecolor="black")'
            plot_title = 'Análise de Outliers - Boxplots para Colunas Numéricas' if plot_type == 'boxplot' else 'Distribuição de Dados - Histogramas para Colunas Numéricas'
            
            codigo_gerado = f"""
import numpy as np

# 1. Identifica colunas numéricas
numerical_cols = df.select_dtypes(include=np.number).columns
num_plots = len(numerical_cols)

if num_plots == 0:
    print("Não há colunas numéricas para plotar.")
else:
    # 2. Calcula o layout dinâmico (max 4 colunas)
    n_cols = min(4, num_plots)
    n_rows = int(np.ceil(num_plots / n_cols))

    # 3. Cria a única figura principal com o layout dinâmico
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten() # Achata para iteração fácil

    # 4. Itera e plota
    for i, col in enumerate(numerical_cols):
        # Usa o eixo (axis) do subplot correto
        {plot_func}
        axes[i].set_title(col, fontsize=10)
        axes[i].tick_params(axis='x', rotation=45)

    # 5. Remove eixos vazios (se existirem)
    for j in range(num_plots, n_rows * n_cols):
        fig.delaxes(axes[j])

    plt.suptitle("{plot_title}", y=1.02, fontsize=14)
    plt.tight_layout()
"""
            conclusoes = f"Os gráficos de {plot_type} foram gerados para visualizar a dispersão dos dados e identificar potenciais problemas de distribuição ou outliers em cada coluna numérica do dataset."
            
            return codigo_gerado, conclusoes

        # --- LÓGICA: GERAÇÃO DE CÓDIGO (RAG - ANÁLISE GERAL) ---
        
        rag_context_str = f"\n\nCONTEXTO ADICIONAL DOS DADOS (RAG):\n{retrieved_context}" if retrieved_context else ""
        historico_conclusoes_str = f"\n\nHISTÓRICO DE ANÁLISE E CONCLUSÕES ANTERIORES:\n{historico_conclusoes}" if historico_conclusoes else ""
        file_context_str = f"\n\nCONTEXTO DO NOME DO ARQUIVO: '{file_context}'."
        
        prompt = f"""
# PERSONA E OBJETIVO PRINCIPAL
Você é um assistente especialista em Análise Exploratória de Dados (E.D.A.) com Pandas.
Sua única função é traduzir uma pergunta em linguagem natural para um código Python.
Você DEVE gerar apenas o código Python.

# CONTEXTO DO DATAFRAME `df`
Esquema do DataFrame:
{schema}
{file_context_str}
{rag_context_str}
{historico_conclusoes_str}

# REGRAS DE GERAÇÃO DE CÓDIGO (MUITO IMPORTANTE)
1.  **Sempre use `df` como o nome do DataFrame.**
2.  **NUNCA gere código para carregar (`pd.read_csv`, `pd.read_excel`, etc.) ou salvar o DataFrame `df`. Ele já está carregado e pronto para uso.**
3.  **Se o resultado for uma tabela de dados (DataFrame), SEMPRE atribua-o a `resultado_df` e imprima `resultado_df` (ex: `print(resultado_df.to_string())`).**
4.  **Para gráficos, use `matplotlib.pyplot` (importado como `plt`). Para gráficos com múltiplos subplots, use `plt.subplots()` com layout dinâmico (`numpy.ceil`).**
5.  **A saída final deve ser APENAS o código Python, sem explicações ou comentários, e JAMAIS inclua qualquer pergunta.**
6.  **EVITE usar zero à esquerda em números decimais inteiros (ex: use '8' em vez de '08') para evitar erro de sintaxe 'octal integers'.**

# PERGUNTA DO USUÁRIO
{pergunta}

# CÓDIGO PYTHON (PANDAS/MATPLOTLIB)
"""
        response = model.generate_content(prompt)
        codigo_gerado = response.text.replace("```python", "").replace("```", "").strip()
        
        # Agente 4: GERA AS CONCLUSÕES APÓS A ANÁLISE
        conclusoes_prompt = f"""
# PERSONA
Você é um analista de dados sênior e seu único trabalho é sintetizar os resultados de uma análise e fornecer conclusões ou insights claros e objetivos para o usuário.

# CONTEXTO
A pergunta do usuário foi: "{pergunta}"
O resultado da análise (Código Python) foi:
{codigo_gerado}

# TAREFA
Com base na pergunta do usuário e nos resultados, forneça uma ou duas frases de conclusão sobre o que foi descoberto. Não mencione o código. Apenas a conclusão.
"""
        conclusoes_response = model.generate_content(conclusoes_prompt)
        conclusoes = conclusoes_response.text

        return codigo_gerado, conclusoes
    except Exception as e:
        return f"Erro ao chamar a API do Gemini: {e}", None

# --- Executor de Código Seguro ---
def executa_codigo_seguro(codigo, df):
    """Executa o código Pandas/Matplotlib gerado em um ambiente isolado."""
    if codigo.startswith("Erro:"):
        return codigo, None, None, None

    output_stream = io.StringIO()
    local_vars = {'df': df, 'pd': pd, 'plt': plt, 'normalize_text': normalize_text, 'np': np} # Adiciona np
    img_bytes = None

    try:
        with contextlib.redirect_stdout(output_stream):
            # Adiciona o df de forma segura para o exec
            local_vars['df'] = df.copy() 
            exec(codigo, {"__builtins__": __builtins__}, local_vars)
        
        # --- Lógica Aprimorada de Captura de Gráfico ---
        # Captura apenas a primeira figura (esperamos que seja a figura com todos os subplots)
        if len(plt.get_fignums()) > 0:
            for fig_num in plt.get_fignums():
                plt.figure(fig_num)
                buf = io.BytesIO()
                
                # Garante que layout está ajustado para subplots
                try:
                    plt.tight_layout()
                except Exception:
                    pass
                    
                plt.savefig(buf, format="png")
                img_bytes = buf.getvalue()
                buf.close()
                plt.close(fig_num)
                break
        
        resultado_texto = output_stream.getvalue().strip()
        resultado_df = local_vars.get('resultado_df')

        # Normaliza Series para DataFrame
        if isinstance(resultado_df, pd.Series):
            resultado_df = resultado_df.reset_index()
            if len(resultado_df.columns) == 2 and 'index' in resultado_df.columns:
                resultado_df.columns = ['Categoria', 'Valor']
        
        # BLOCO CRÍTICO: GARANTE QUE TODO TEXTO SEJA CONVERTIDO EM DATAFRAME (TABELA)
        if resultado_df is None and resultado_texto and not img_bytes:
            # Cria um DataFrame de 1x1 com a resposta textual
            resultado_df = pd.DataFrame({'INFORMAÇÃO': [resultado_texto]})
            # Limpa o resultado_texto para que o Streamlit priorize resultado_df
            resultado_texto = ""

        return resultado_texto, resultado_df, None, img_bytes

    except Exception as e:
        error_message = f"Erro ao executar o código gerado pela IA:\n\n{e}\n\nCódigo que falhou:\n```python\n{codigo}\n```"
        return error_message, None, error_message, None

# --- Agente 3: Formatação da Apresentação ---
def agente3_formatar_apresentacao(resultado_texto, resultado_df, pergunta, img_bytes):
    """Gera o relatório em PDF (ReportLab)."""
    
    pdf_bytes = None
    final_text_output = resultado_texto

    # Usa o DataFrame para texto se estiver disponível (para PDF)
    if resultado_df is not None and not resultado_df.empty:
        final_text_output = resultado_df.to_markdown(index=resultado_df.index.name is not None)
    
    try:
        pdf_output_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_output_buffer, pagesize=A4)
        width, height = A4
        
        c.setFont("Helvetica-Bold", 14)
        c.drawString(inch, height - inch, "Relatório de Análise de Dados")
        
        c.setFont("Helvetica", 12)
        textobject = c.beginText(inch, height - inch - 20)
        textobject.textLines(f"Pergunta: {pergunta}")
        c.drawText(textobject)
        
        y_pos = height - inch - 50

        if resultado_df is not None and not resultado_df.empty:
            
            # Adiciona o índice como primeira coluna se o DataFrame tiver um nome de índice definido
            if resultado_df.index.name is not None:
                data = [
                    [resultado_df.index.name] + resultado_df.columns.tolist()
                ] + [
                    [str(idx)] + row for idx, row in zip(resultado_df.index.tolist(), resultado_df.values.astype(str).tolist())
                ]
            else:
                data = [resultado_df.columns.tolist()] + resultado_df.values.astype(str).tolist()

            # Limita a largura da tabela
            max_cols = 10 
            data = [row[:max_cols] for row in data]
            
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            table_width, table_height = table.wrapOn(c, width, height)
            y_pos -= table_height + 20
            
            if y_pos < inch:
                c.showPage()
                y_pos = height - inch - 20
            
            table.drawOn(c, inch, y_pos)
            y_pos -= 20
        
        if img_bytes:
            image_reader = ImageReader(io.BytesIO(img_bytes))
            img_width, img_height = image_reader.getSize()
            
            aspect_ratio = img_width / img_height
            max_width = width - 2 * inch
            
            img_width = max_width
            img_height = img_width / aspect_ratio
            
            y_pos -= img_height + 20
            
            if y_pos < inch:
                c.showPage()
                y_pos = height - inch - 20
            
            c.drawImage(image_reader, inch, y_pos, width=img_width, height=img_height)

        c.save()
        pdf_bytes = pdf_output_buffer.getvalue()

    except Exception as e:
        pdf_bytes = None

    return final_text_output, img_bytes, pdf_bytes

# --- Streamlit UI ---
st.title("Análise Exploratória de Dados (EDA) com Gemini e RAG")
st.markdown("---")


# --- Configuração da Sidebar ---
with st.sidebar:
    st.header("Configurações")
    with st.form("api_key_form"):
        api_key_input = st.text_input("Cole sua API Key do Gemini aqui", value=st.session_state['gemini_api_key'], type="password", key="gemini_api_key_input_form")
        submitted = st.form_submit_button("Salvar API Key")
        if submitted:
            st.session_state['gemini_api_key'] = api_key_input
            st.success("API Key salva com sucesso!")

    st.markdown("---")
    st.info("1. Faça o upload do arquivo **ZIP**.")
    st.info("2. Clique em **'Listar Arquivos'** para identificação.")
    st.info("3. Escolha o arquivo e clique em **'Analisar'**.")
    st.info("4. Faça sua pergunta de EDA.")

# --- Seção 1: Upload e Seleção de Dados ---
st.header("1. Upload e Seleção de Dados")
zipfile_input = st.file_uploader("Selecione o arquivo ZIP com o(s) arquivo(s) de dados (CSV, XLSX, TXT)", type=["zip"])

if st.button("Listar Arquivos no ZIP"):
    if zipfile_input is not None:
        st.session_state['zip_bytes'] = zipfile_input.getvalue()
        st.session_state['zip_hash'] = hashlib.md5(st.session_state['zip_bytes']).hexdigest()
        
        # Reseta estados importantes
        st.session_state['selected_file_name'] = None 
        st.session_state['df'] = None 
        st.session_state['conclusoes_historico'] = "" 
        st.session_state['processed_percentage'] = 0
        st.session_state['available_files'] = []
        st.session_state['file_options_map'] = {}
        st.session_state['faiss_index'] = None
        st.session_state['documents'] = []


        with st.spinner("Analisando arquivos e gerando contexto com Gemini..."):
            file_info_list = agente1_identifica_arquivos(st.session_state['zip_bytes'])
            
            if not file_info_list:
                st.error("O ZIP não contém arquivos CSV, XLSX ou TXT válidos.")
            else:
                st.session_state['available_files'] = file_info_list
                
                file_context_map = agente1_interpreta_contexto_arquivo(st.session_state.get('gemini_api_key'), file_info_list)
                
                options = {}
                for info in file_info_list:
                    context = file_context_map.get(info['name'], "Contexto não gerado.")
                    options[info['name']] = f"**{info['name']}** - {context}"
                
                st.session_state['file_options_map'] = options
                st.success(f"Encontrados {len(file_info_list)} arquivos de dados. Escolha qual analisar.")

if st.session_state['available_files']:
    
    st.markdown("---")
    st.subheader("Selecione o Arquivo para Análise:")
    
    options_list = list(st.session_state['file_options_map'].keys())
    
    if not options_list:
        st.error("Nenhum arquivo elegível encontrado após a identificação.")
    else:
        display_options = [st.session_state['file_options_map'][key] for key in options_list]
        
        default_index = 0
        try:
            if st.session_state['selected_file_name'] in options_list:
                default_index = options_list.index(st.session_state['selected_file_name'])
        except:
             default_index = 0

        selected_display = st.radio(
            "Arquivos Encontrados (Nome - Contexto Inferido):",
            options=display_options,
            index=default_index,
            key="file_selection_radio"
        )
        
        selected_file_name = options_list[display_options.index(selected_display)]
        st.session_state['selected_file_name'] = selected_file_name
        
        selected_file_info = next((info for info in st.session_state['available_files'] if info["name"] == selected_file_name), None)
        
        if st.button(f"Analisar Arquivo: {selected_file_name}") and selected_file_info:
            
            expected_num_cols = selected_file_info['num_cols']
            
            # --- INÍCIO DO PROCESSO DE CARGA/CHUNKED (RAG) ---
            
            st.info(f"Tentando carregar progresso anterior para **{selected_file_name}**...")
            
            # Tenta carregar o progresso anterior
            df_loaded, index_loaded, docs_loaded, lines_loaded_processed = load_progress(st.session_state['zip_hash'], selected_file_name)
            
            st.session_state['df'] = df_loaded
            st.session_state['faiss_index'] = index_loaded
            st.session_state['documents'] = docs_loaded
            st.session_state['current_chunk_start'] = lines_loaded_processed # Onde deve continuar o chunking
            
            # Tenta obter o total de linhas real do arquivo
            total_lines_file = 0
            try:
                ext = selected_file_info['extension']
                if ext == '.csv' or ext == '.txt':
                    with zipfile.ZipFile(io.BytesIO(st.session_state['zip_bytes']), "r") as z:
                        with z.open(selected_file_name, 'r') as file_in_zip:
                            # Subtrai 1 para o cabeçalho
                            total_lines_file = sum(1 for line in io.TextIOWrapper(file_in_zip, encoding='utf-8', errors='ignore')) - 1 
                else:
                    # Para XLSX, apenas usamos uma estimativa inicial alta
                    total_lines_file = CHUNK_SIZE * 50 
                    
            except Exception as e:
                total_lines_file = CHUNK_SIZE * 10 
            
            # Define o total de linhas real do arquivo (ou o que foi processado se for maior que a estimativa)
            st.session_state['total_lines'] = max(total_lines_file, lines_loaded_processed)
            
            st.session_state['file_name_context'] = normalize_text(os.path.splitext(selected_file_name)[0].upper().replace('_', ' ').replace('-', ' '))

            # Verifica se o carregamento foi completo ou se precisa continuar
            if lines_loaded_processed > 0 and lines_loaded_processed >= total_lines_file:
                st.session_state['df'] = agente_limpeza_dados(st.session_state['df'])
                st.session_state['processed_percentage'] = 100
                st.success(f"Processamento de **{selected_file_name}** concluído (total de linhas: {len(st.session_state['df'])}).")
                progress_bar = st.progress(1.0, text="Processamento finalizado. A ferramenta está pronta para uso!")
                st.rerun() 
            
            # Se o carregamento parcial ocorreu, precisamos continuar
            elif lines_loaded_processed > 0 and lines_loaded_processed < total_lines_file:
                st.info(f"Progresso parcial encontrado ({lines_loaded_processed} linhas). Continuaremos o processamento para as {total_lines_file - lines_loaded_processed} linhas restantes.")
                st.session_state['df_columns'] = st.session_state['df'].columns # Garante que as colunas sejam mantidas
                st.session_state['df'] = agente_limpeza_dados(st.session_state['df']) # Limpa a parte já carregada
            
            # --- INÍCIO DO NOVO PROCESSAMENTO (Se o carregamento falhou ou é a primeira vez) ---
            else:
                st.info(f"Iniciando novo processamento para **{selected_file_name}** ({st.session_state['total_lines']} linhas estimadas)...")
                st.session_state['df'] = None
                st.session_state['faiss_index'] = None
                st.session_state['documents'] = []
                st.session_state['conclusoes_historico'] = ""
                st.session_state['df_columns'] = None
                st.session_state['processed_percentage'] = 0
                st.session_state['cleaned_status'] = {}
                st.session_state['current_chunk_start'] = 0
                lines_loaded_processed = 0


            # Loop de processamento de chunks
            progress_bar = st.progress(lines_loaded_processed / st.session_state['total_lines'], 
                                       text=f"Criando embeddings e índice RAG... {lines_loaded_processed}/{st.session_state['total_lines']} linhas...")
            
            start_row = lines_loaded_processed
            
            while start_row < st.session_state['total_lines'] or start_row == 0:
                
                chunk_processed, msg = agente1_processa_arquivo_chunk(
                    st.session_state['zip_bytes'], 
                    selected_file_name, 
                    start_row, 
                    CHUNK_SIZE, 
                    st.session_state['df_columns'],
                    expected_num_cols
                )
                
                if chunk_processed is not None:
                    
                    # 1. Aplica limpeza e concatena
                    chunk_processed = agente_limpeza_dados(chunk_processed)
                    
                    if st.session_state['df'] is None:
                        st.session_state['df'] = chunk_processed
                        st.session_state['df_columns'] = chunk_processed.columns
                        # Re-calcula o número total de colunas esperado
                        expected_num_cols = len(st.session_state['df_columns'])
                    else:
                        # Garante que as colunas do chunk coincidam com o DF principal
                        if len(chunk_processed.columns) == len(st.session_state['df_columns']):
                            chunk_processed.columns = st.session_state['df_columns']
                        st.session_state['df'] = pd.concat([st.session_state['df'], chunk_processed], ignore_index=True)
                    
                    # 2. Cria índice RAG para o chunk
                    create_faiss_index_for_chunk(chunk_processed)
                    
                    # 3. Atualiza progresso
                    start_row += len(chunk_processed)
                    st.session_state['current_chunk_start'] = start_row
                    
                    # Recalibra o total de linhas se necessário
                    if len(chunk_processed) < CHUNK_SIZE and start_row < st.session_state['total_lines']:
                         st.session_state['total_lines'] = start_row
                         
                    progress_value = min(start_row / st.session_state['total_lines'], 1.0) if st.session_state['total_lines'] > 0 else 1.0
                    st.session_state['processed_percentage'] = progress_value * 100
                    
                    progress_bar.progress(progress_value, 
                                          text=f"Criando embeddings e índice RAG... {start_row}/{st.session_state['total_lines']} linhas - {st.session_state['processed_percentage']:.1f}%")
                    
                    save_progress(st.session_state['zip_hash'], st.session_state['df'], st.session_state['faiss_index'], st.session_state['documents'], st.session_state['total_lines'])
                    
                    # Condição de parada (processou o último chunk)
                    if len(chunk_processed) < CHUNK_SIZE:
                        st.session_state['total_lines'] = start_row # Fixa o total de linhas
                        break
                        
                else:
                    if "todos os lotes concluído" in msg:
                        st.session_state['total_lines'] = start_row # Fixa o total de linhas
                        break
                    st.error(msg)
                    break
            
            if st.session_state['df'] is not None and len(st.session_state['df']) > 0:
                st.session_state['total_lines'] = len(st.session_state['df'])
                st.success(f"Processamento de **{selected_file_name}** concluído! Total de linhas carregadas: {len(st.session_state['df'])}")
                progress_bar.progress(1.0, text="Processamento finalizado. A ferramenta está pronta para uso!")
                st.session_state['processed_percentage'] = 100
                st.rerun() 
            else:
                progress_bar.empty()
                st.error("Falha ao carregar o arquivo. Verifique se o formato está correto.")

st.markdown("---")

# --- Seção 2: Consulta à IA ---

st.header("2. Consultar a IA")

if st.session_state['df'] is None or st.session_state['processed_percentage'] < 5.0:
    st.info("Aguardando o processamento do arquivo para habilitar a consulta.")
else:
    if st.session_state['file_name_context'] and st.session_state['selected_file_name']:
        st.info(f"Analisando: **{st.session_state['selected_file_name']}**. Contexto: **{st.session_state['file_name_context']}**.")

    pergunta = st.text_area(
        "Pergunte em português sobre os dados:",
        value=st.session_state.get('user_query_input_widget', ""),
        placeholder="Ex: Qual o tipo de cada coluna? Me dê as estatísticas descritivas. Há correlação entre V1 e V2? Gere um boxplot para outliers.",
        height=100,
        key="user_query_input_widget"
    )
    
    # Botões de Ação
    col1_icons, col2_icons, col3_icons, col4_icons, col5_icons = st.columns([1, 1, 1, 1, 4])
    
    if col1_icons.button("Consultar (🔎)"):
        st.session_state['consultar_ia'] = True

    # Botões de display para ativar após a consulta (só aparecem após a primeira consulta)
    if st.session_state.get('codigo_gerado'):
        with col2_icons:
            if st.button("Gráfico (📊)"):
                st.session_state['habilitar_grafico'] = True
        with col3_icons:
            if st.button("Código (✍️)"):
                st.session_state['exibir_codigo'] = True
        with col4_icons:
            if st.button("PDF (📄)"):
                st.session_state['gerar_pdf'] = True

    st.markdown("---")
    
    # --- Lógica de Execução da Consulta (BLOBO ATUALIZADO) ---
    if 'consultar_ia' in st.session_state and st.session_state['consultar_ia']:
        st.session_state['consultar_ia'] = False
        
        if not st.session_state.get('gemini_api_key'):
            st.error("Por favor, insira e salve sua API Key do Gemini na barra lateral.")
        elif st.session_state['faiss_index'] is None or st.session_state['faiss_index'].ntotal == 0:
            st.warning("O índice RAG não foi criado. Por favor, processe o arquivo (clique em 'Analisar Arquivo' e aguarde o progresso).")
        else:
            pergunta_original = pergunta # Captura a pergunta original do widget
            
            # --- NOVA ETAPA DE CLARIFICAÇÃO ---
            with st.spinner("Clarificando sua pergunta e corrigindo possíveis erros de digitação..."):
                pergunta_clarificada = agente0_clarifica_pergunta(pergunta_original, st.session_state['gemini_api_key'])
            
            pergunta_para_ia = pergunta_clarificada # A partir daqui, usa a versão limpa
            
            # Exibe a correção se ela ocorreu
            if pergunta_para_ia != pergunta_original:
                 st.warning(f"Sua consulta foi clarificada para: **{pergunta_para_ia}**")
            # --- FIM DA NOVA ETAPA ---
            
            st.info(f"Análise realizada sobre **{st.session_state['processed_percentage']:.1f}%** dos dados já processados (total de **{len(st.session_state['df'])}** linhas).")
            
            with st.spinner("Gerando código e analisando dados..."):
                df_to_use = st.session_state['df']
                faiss_index = st.session_state['faiss_index']
                documents = st.session_state['documents']
                api_key = st.session_state['gemini_api_key']

                # 1. Recupera o Contexto (RAG) - USANDO A PERGUNTA CLARIFICADA
                retrieved_context = retrieve_context(pergunta_para_ia, faiss_index, documents)
                
                # 2. Gera Código e Conclusão - USANDO A PERGUNTA CLARIFICADA
                codigo_gerado, conclusoes = agente2_gera_codigo_pandas_eda(
                    pergunta_para_ia, 
                    api_key, 
                    df_to_use, 
                    retrieved_context, 
                    st.session_state['conclusoes_historico'],
                    st.session_state['file_name_context'] 
                )
                
                if conclusoes:
                    # Adiciona a nova conclusão ao histórico
                    st.session_state['conclusoes_historico'] += f"\n- {conclusoes}"
                
                st.session_state['codigo_gerado'] = codigo_gerado
                
                # 3. Executa o Código
                if codigo_gerado.startswith("Erro:"):
                    st.error(codigo_gerado)
                else:
                    resultado_texto, resultado_df, erro_execucao, img_bytes = executa_codigo_seguro(codigo_gerado, df_to_use)
                    
                    st.session_state['resultado_texto'] = resultado_texto
                    st.session_state['resultado_df'] = resultado_df
                    st.session_state['erro_execucao'] = erro_execucao
                    st.session_state['img_bytes'] = img_bytes
                    
                    if erro_execucao:
                        st.error(erro_execucao)
                    else:
                        st.subheader("Resultado da Análise:")
                        
                        # Exibe Tabela (Corrigido para evitar truncamento em colunas longas)
                        if resultado_df is not None and not resultado_df.empty:
                            
                            # Hack para colunas longas (Texto completo)
                            if 'INFORMAÇÃO' in resultado_df.columns:
                                st.markdown("##### Informação Detalhada (Texto Completo):")
                                
                                # Cria a tabela Markdown (Título e Corpo)
                                table_markdown = "| | INFORMAÇÃO |\n"
                                table_markdown += "| :--- | :--- |\n"
                                
                                # Adiciona as linhas do DataFrame
                                for index, row in resultado_df.iterrows():
                                    # Usa o índice como a primeira coluna e o texto como a segunda
                                    index_display = index if resultado_df.index.name is None else row.name
                                    table_markdown += f"| **{index_display}** | {row['INFORMAÇÃO']} |\n"
                                
                                st.markdown(table_markdown)
                                
                            else:
                                # Usa o st.dataframe normal para colunas numéricas/curtas
                                column_config = {col: st.column_config.Column(
                                    width="large",
                                    help="Descrição"
                                ) for col in resultado_df.columns}

                                if resultado_df.index.name:
                                    column_config[resultado_df.index.name] = st.column_config.TextColumn(
                                        width="small",
                                        help="Tipo/Índice"
                                    )
                                st.dataframe(resultado_df, use_container_width=True, column_config=column_config)
                        
                        # Exibe Gráfico (se houver)
                        if img_bytes and 'habilitar_grafico' not in st.session_state:
                            st.subheader("Gráfico Gerado:")
                            st.image(img_bytes, caption="Gráfico da Análise", use_container_width=True)
    
    # --- Lógica de Exibição dos Botões Secundários ---
    
    if 'exibir_codigo' in st.session_state and st.session_state['exibir_codigo']:
        st.session_state['exibir_codigo'] = False
        if st.session_state.get('codigo_gerado'):
            st.subheader("Cógido Python Gerado:")
            st.code(st.session_state['codigo_gerado'], language='python')
        else:
            st.warning("Nenhum código gerado. Faça uma consulta primeiro.")

    if 'habilitar_grafico' in st.session_state and st.session_state['habilitar_grafico']:
        st.session_state['habilitar_grafico'] = False
        if st.session_state.get('img_bytes'):
            st.subheader("Gráfico Gerado:")
            st.image(st.session_state['img_bytes'], caption="Gráfico da Análise", use_container_width=True)
        else:
            st.warning("Nenhum gráfico gerado na última consulta.")

    if 'gerar_pdf' in st.session_state and st.session_state['gerar_pdf']:
        st.session_state['gerar_pdf'] = False
        if st.session_state.get('codigo_gerado'):
            with st.spinner("Gerando PDF..."):
                _, _, pdf_bytes = agente3_formatar_apresentacao(st.session_state['resultado_texto'], st.session_state.get('resultado_df'), pergunta, st.session_state.get('img_bytes'))
                
                if pdf_bytes:
                    st.subheader("Download do Relatório:")
                    st.download_button(
                        label="Baixar Relatório em PDF",
                        data=pdf_bytes,
                        file_name="relatorio_eda.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.warning("Não foi possível gerar o PDF. Verifique se as bibliotecas (ReportLab) estão instaladas.")
        else:
            st.warning("Por favor, execute uma consulta e analise os dados primeiro para gerar o PDF.")

st.markdown("---")

# --- Seção 3: Conclusões do Agente ---
st.header("3. Conclusões do Agente")
if st.session_state['conclusoes_historico']:
    st.markdown(st.session_state['conclusoes_historico'])
else:
    st.info("As conclusões da análise aparecerão aqui após a primeira consulta.")

st.markdown("---")
st.markdown("Adaptado por Rafael - grupo TENKAI")
