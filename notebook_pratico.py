#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Notebook Prático: Expressões Regulares para Processos Jurídicos
Curso: Análise de Processos sobre Crédito Consignado
"""

import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configuração para visualização
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")

print("🏛️  ANÁLISE DE PROCESSOS JURÍDICOS COM REGEX")
print("=" * 50)

# =============================================================================
# 1. IMPORTAÇÃO E ESTRUTURA DOS DADOS
# =============================================================================

def load_and_explore_data(filepath):
    """Carrega e explora a estrutura do dataset"""
    
    print("📂 CARREGANDO DATASET...")
    
    # Carregar dados
    df = pd.read_csv(filepath, delimiter='|', encoding='utf-8')
    
    print(f"✅ Dataset carregado: {len(df)} registros")
    print(f"📊 Colunas: {list(df.columns)}")
    
    # Estatísticas básicas
    print("\n📈 ESTRUTURA DOS DADOS:")
    for col in df.columns:
        non_null = df[col].notna().sum()
        max_length = df[col].astype(str).str.len().max()
        print(f"  {col}: {non_null}/{len(df)} preenchidos (max: {max_length} chars)")
    
    # Amostra dos tipos de ação
    print("\n⚖️  TIPOS DE AÇÃO MAIS COMUNS:")
    acao_counts = df['ds_Acao_Judicial'].value_counts().head(5)
    for acao, count in acao_counts.items():
        print(f"  {count:4d}x: {acao}")
    
    return df

# Carregar dados (substitua pelo caminho correto)
# df = load_and_explore_data('dataset_clinica20252.csv')

# =============================================================================
# 2. LIMPEZA E NORMALIZAÇÃO
# =============================================================================

def clean_text(text):
    """Limpa e normaliza texto jurídico"""
    
    if pd.isna(text) or text == '':
        return ""
    
    text = str(text)
    
    # Correções de encoding comuns
    encoding_fixes = {
        'Ã¡': 'á', 'Ã ': 'à', 'Ã£': 'ã', 'Ã©': 'é', 'Ã­': 'í', 
        'Ã³': 'ó', 'Ãº': 'ú', 'Ã§': 'ç', 'Ãª': 'ê', 'Ã´': 'ô',
        'Ã\x81': 'Á', 'Ã\x80': 'À', 'Ã\x83': 'Ã', 'Ã\x89': 'É',
        'Ã\x8d': 'Í', 'Ã\x93': 'Ó', 'Ã\x9a': 'Ú', 'Ã\x87': 'Ç'
    }
    
    for wrong, correct in encoding_fixes.items():
        text = text.replace(wrong, correct)
    
    # Normalização de espaços e quebras de linha
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def apply_cleaning(df):
    """Aplica limpeza ao dataframe"""
    
    print("🧹 APLICANDO LIMPEZA DOS DADOS...")
    
    # Limpar campos textuais principais
    text_fields = ['ds_fatos', 'ds_Pedidos', 'ds_Qualificacao']
    
    for field in text_fields:
        if field in df.columns:
            df[f'{field}_clean'] = df[field].apply(clean_text)
            print(f"  ✅ {field} limpo")
    
    return df

# Aplicar limpeza
# df = apply_cleaning(df)

# =============================================================================
# 3. IDENTIFICAÇÃO DE CRÉDITO CONSIGNADO
# =============================================================================

def build_consignado_patterns():
    """Constrói padrões regex para identificar crédito consignado"""
    
    patterns = {
        'consignado_direto': [
            r'cr[eé]dito\s+consignado',
            r'empr[eé]stimo\s+consignado',
            r'consigna[çc][aã]o'
        ],
        
        'desconto_beneficio': [
            r'desconto\s+(?:em\s+|na\s+)?folha',
            r'margem\s+consign[aá]vel',
            r'benef[íi]cio\s+(?:do\s+)?INSS.*desconto',
            r'aposentadoria.*empr[eé]stimo',
            r'desconto\s+(?:indevido|fraudulento).*benef[íi]cio'
        ],
        
        'indicadores_fraude': [
            r'empr[eé]stimo.*(?:n[aã]o\s+)?contrat(?:ou|ado)',
            r'desconto.*sem\s+(?:conhecimento|autoriza[çc][aã]o)',
            r'(?:fraude|fraudulent[oa]).*empr[eé]stimo',
            r'(?:indevido|ilegal).*desconto'
        ]
    }
    
    return patterns

def is_credito_consignado(texto, patterns):
    """Identifica se o processo é sobre crédito consignado"""
    
    if not texto:
        return False, []
    
    texto_lower = texto.lower()
    matched_patterns = []
    
    # Verificar cada categoria de padrões
    for categoria, pattern_list in patterns.items():
        for pattern in pattern_list:
            if re.search(pattern, texto_lower):
                matched_patterns.append(f"{categoria}: {pattern}")
    
    return len(matched_patterns) > 0, matched_patterns

def analyze_consignado_cases(df):
    """Analisa casos de crédito consignado no dataset"""
    
    print("🔍 IDENTIFICANDO PROCESSOS DE CRÉDITO CONSIGNADO...")
    
    patterns = build_consignado_patterns()
    
    # Aplicar identificação
    results = []
    for idx, row in df.iterrows():
        texto_completo = f"{row['ds_fatos_clean']} {row['ds_Pedidos_clean']}"
        is_consig, matched = is_credito_consignado(texto_completo, patterns)
        results.append({
            'is_consignado': is_consig,
            'patterns_matched': matched,
            'confidence': len(matched)
        })
    
    # Adicionar resultados ao dataframe
    results_df = pd.DataFrame(results)
    df['is_consignado'] = results_df['is_consignado']
    df['patterns_matched'] = results_df['patterns_matched']
    df['confidence'] = results_df['confidence']
    
    # Estatísticas
    total_consignado = df['is_consignado'].sum()
    print(f"  ✅ Identificados: {total_consignado}/{len(df)} ({100*total_consignado/len(df):.1f}%)")
    
    return df

# =============================================================================
# 4. EXTRAÇÃO DE VALORES MONETÁRIOS
# =============================================================================

def extract_monetary_values(texto):
    """Extrai valores monetários de diferentes formatos"""
    
    if not texto:
        return []
    
    valores = []
    
    # Padrão 1: R$ 1.234,56 ou R$1234,56
    pattern_currency = r'R\$\s*(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)'
    matches = re.findall(pattern_currency, texto)
    for match in matches:
        try:
            valor = float(match.replace('.', '').replace(',', '.'))
            valores.append(('moeda', valor))
        except:
            continue
    
    # Padrão 2: Valores por extenso com parênteses
    pattern_extenso = r'(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)\s*\([^)]*reais?\)'
    matches = re.findall(pattern_extenso, texto)
    for match in matches:
        try:
            valor = float(match.replace('.', '').replace(',', '.'))
            valores.append(('extenso', valor))
        except:
            continue
    
    # Padrão 3: Salários mínimos
    pattern_salario = r'(\d+)\s*(?:\([^)]+\))?\s*sal[aá]rios?[\s-]m[íi]nimos?'
    matches = re.findall(pattern_salario, texto, re.IGNORECASE)
    for match in matches:
        try:
            qtd = int(match)
            valor = qtd * 1518.00  # Salário mínimo 2025
            valores.append(('salario_minimo', valor))
        except:
            continue
    
    # Padrão 4: Percentuais sobre valores
    pattern_percent = r'(\d+(?:,\d+)?)\s*%\s*(?:sobre|de|do)\s*(?:valor|montante)'
    # Este padrão precisa de contexto adicional para calcular o valor real
    
    return valores

def extract_indenization_values(df):
    """Extrai valores de indenização dos pedidos"""
    
    print("💰 EXTRAINDO VALORES DE INDENIZAÇÃO...")
    
    all_values = []
    
    for idx, row in df.iterrows():
        # Focar nos pedidos para indenização
        pedidos = row['ds_Pedidos_clean']
        
        # Extrair valores
        valores = extract_monetary_values(pedidos)
        
        # Filtrar valores relevantes (acima de R$ 100 e abaixo de R$ 1 milhão)
        valores_filtrados = [v for tipo, v in valores if 100 <= v <= 1000000]
        
        all_values.append({
            'cd_atendimento': row['cd_atendimento'],
            'valores_brutos': valores,
            'valores_filtrados': valores_filtrados,
            'maior_valor': max(valores_filtrados) if valores_filtrados else np.nan,
            'total_valores': sum(valores_filtrados) if valores_filtrados else np.nan,
            'qtd_valores': len(valores_filtrados)
        })
    
    # Converter para DataFrame e juntar
    values_df = pd.DataFrame(all_values)
    df = df.merge(values_df[['cd_atendimento', 'maior_valor', 'total_valores', 'qtd_valores']], 
                  on='cd_atendimento', how='left')
    
    # Estatísticas
    with_values = df['qtd_valores'].gt(0).sum()
    print(f"  ✅ Processos com valores: {with_values}/{len(df)} ({100*with_values/len(df):.1f}%)")
    print(f"  💵 Valor médio: R$ {df['maior_valor'].mean():.2f}")
    print(f"  📊 Mediana: R$ {df['maior_valor'].median():.2f}")
    
    return df

# =============================================================================
# 5. IDENTIFICAÇÃO DE EMPRESAS/BANCOS
# =============================================================================

def extract_companies(texto):
    """Extrai nomes de empresas e bancos dos textos"""
    
    if not texto:
        return []
    
    empresas = []
    
    # Padrões específicos para bancos brasileiros
    banks_patterns = [
        r'BANCO\s+(?:DO\s+)?BRADESCO\s*S\.?/?A\.?',
        r'BANCO\s+SANTANDER\s*(?:BRASIL\s*)?S\.?/?A\.?',
        r'CAIXA\s+ECON[ÔO]MICA\s+FEDERAL',
        r'BANCO\s+(?:DO\s+)?BRASIL\s*S\.?/?A\.?',
        r'ITA[ÚU]\s*(?:UNIBANCO\s*)?S\.?/?A\.?',
        r'BANCO\s+INTER\s*S\.?/?A\.?',
        r'BANCO\s+SAFRA\s*S\.?/?A\.?',
        r'BANCO\s+BTG\s+PACTUAL\s*S\.?/?A\.?',
        r'BANCO\s+ORIGINAL\s*S\.?/?A\.?',
        r'BANCO\s+PAN\s*S\.?/?A\.?'
    ]
    
    # Buscar bancos
    for pattern in banks_patterns:
        matches = re.findall(pattern, texto, re.IGNORECASE)
        empresas.extend([match.upper().strip() for match in matches])
    
    # Padrão genérico para outras empresas
    # Busca por palavras em CAPS seguidas de S.A., LTDA, etc.
    pattern_empresa = r'([A-Z][A-Z\s&]{3,}(?:S\.?/?A\.?|LTDA\.?|S\.?S\.?|EIRELI))'
    matches_empresa = re.findall(pattern_empresa, texto)
    
    for match in matches_empresa:
        empresa_clean = re.sub(r'\s+', ' ', match).strip()
        if len(empresa_clean) > 5:  # Filtrar strings muito curtas
            empresas.append(empresa_clean)
    
    # Remover duplicatas e retornar
    return list(set(empresas))

def analyze_companies(df):
    """Analisa empresas mencionadas nos processos"""
    
    print("🏢 IDENTIFICANDO EMPRESAS PROCESSADAS...")
    
    # Extrair empresas da qualificação (onde geralmente estão os réus)
    df['empresas'] = df['ds_Qualificacao'].apply(extract_companies)
    
    # Estatísticas
    with_companies = df['empresas'].apply(len).gt(0).sum()
    print(f"  ✅ Processos com empresas identificadas: {with_companies}/{len(df)}")
    
    # Ranking das empresas mais processadas
    all_companies = [emp for lista in df['empresas'] for emp in lista]
    company_counts = Counter(all_companies)
    
    print("  🏆 EMPRESAS MAIS PROCESSADAS:")
    for empresa, count in company_counts.most_common(10):
        print(f"    {count:3d}x: {empresa}")
    
    return df

# =============================================================================
# 6. VALIDAÇÃO E MÉTRICAS
# =============================================================================

def validate_model(df, sample_size=20):
    """Valida os resultados em uma amostra manual"""
    
    print("✅ VALIDAÇÃO DO MODELO...")
    
    # Selecionar amostra estratificada
    consignado_sample = df[df['is_consignado']].sample(n=min(sample_size//2, df['is_consignado'].sum()))
    non_consignado_sample = df[~df['is_consignado']].sample(n=sample_size//2)
    
    validation_sample = pd.concat([consignado_sample, non_consignado_sample])
    
    print(f"  📝 Amostra para validação: {len(validation_sample)} casos")
    print("  💡 Revise manualmente os casos abaixo:")
    print("=" * 60)
    
    for idx, row in validation_sample.iterrows():
        print(f"\n🔍 CASO: {row['cd_atendimento']}")
        print(f"PREDIÇÃO: {'✅ CONSIGNADO' if row['is_consignado'] else '❌ NÃO CONSIGNADO'}")
        print(f"CONFIANÇA: {row['confidence']} padrões")
        
        if row['is_consignado']:
            print(f"PADRÕES: {', '.join(row['patterns_matched'])}")
        
        # Mostrar trecho relevante dos fatos
        fatos = row['ds_fatos_clean']
        if len(fatos) > 200:
            fatos = fatos[:200] + "..."
        print(f"FATOS: {fatos}")
        
        print("-" * 60)

def generate_statistics(df):
    """Gera estatísticas descritivas do dataset processado"""
    
    print("📊 ESTATÍSTICAS FINAIS...")
    
    # Estatísticas gerais
    total_processos = len(df)
    processos_consignado = df['is_consignado'].sum()
    processos_com_valores = df['qtd_valores'].gt(0).sum()
    
    print(f"  📈 Total de processos: {total_processos}")
    print(f"  ⚖️  Crédito consignado: {processos_consignado} ({100*processos_consignado/total_processos:.1f}%)")
    print(f"  💰 Com valores extraídos: {processos_com_valores} ({100*processos_com_valores/total_processos:.1f}%)")
    
    # Estatísticas de valores
    if processos_com_valores > 0:
        print(f"\n💵 ANÁLISE DE VALORES:")
        print(f"  Valor médio: R$ {df['maior_valor'].mean():.2f}")
        print(f"  Mediana: R$ {df['maior_valor'].median():.2f}")
        print(f"  Valor mínimo: R$ {df['maior_valor'].min():.2f}")
        print(f"  Valor máximo: R$ {df['maior_valor'].max():.2f}")
    
    return df

# =============================================================================
# 7. CASOS PARA LLM
# =============================================================================

def identify_llm_candidates(df):
    """Identifica casos que precisam de análise por LLM"""
    
    print("🤖 IDENTIFICANDO CASOS PARA LLM...")
    
    llm_cases = []
    
    for idx, row in df.iterrows():
        needs_llm = False
        reasons = []
        
        # Critério 1: Texto muito longo (pode ter informações escondidas)
        if len(row['ds_fatos_clean']) > 3000:
            needs_llm = True
            reasons.append("texto_muito_longo")
        
        # Critério 2: Baixa confiança na classificação
        if row['is_consignado'] and row['confidence'] <= 1:
            needs_llm = True
            reasons.append("baixa_confianca_consignado")
        
        # Critério 3: Nenhum valor encontrado mas indica indenização
        indenizacao_terms = r'indeniza[çc][aã]o|danos?\s+morais?|repara[çc][aã]o'
        if (row['qtd_valores'] == 0 and 
            re.search(indenizacao_terms, row['ds_Pedidos_clean'], re.IGNORECASE)):
            needs_llm = True
            reasons.append("valores_nao_identificados")
        
        # Critério 4: Múltiplas empresas (caso complexo)
        if len(row['empresas']) > 2:
            needs_llm = True
            reasons.append("multiplas_empresas")
        
        # Critério 5: Linguagem muito técnica
        termos_tecnicos = ['usucapião', 'exceptio', 'sub-rogação', 'cessão', 'novação']
        if any(termo in row['ds_fatos_clean'].lower() for termo in termos_tecnicos):
            needs_llm = True
            reasons.append("linguagem_muito_tecnica")
        
        if needs_llm:
            llm_cases.append({
                'cd_atendimento': row['cd_atendimento'],
                'reasons': reasons,
                'priority': len(reasons)  # Mais razões = maior prioridade
            })
    
    # Ordenar por prioridade
    llm_cases.sort(key=lambda x: x['priority'], reverse=True)
    
    print(f"  🎯 Casos para LLM: {len(llm_cases)}/{len(df)} ({100*len(llm_cases)/len(df):.1f}%)")
    
    # Mostrar distribuição dos motivos
    all_reasons = [reason for case in llm_cases for reason in case['reasons']]
    reason_counts = Counter(all_reasons)
    
    print("  📋 MOTIVOS PARA LLM:")
    for reason, count in reason_counts.most_common():
        print(f"    {count:3d}x: {reason}")
    
    return llm_cases

# =============================================================================
# 8. VISUALIZAÇÕES
# =============================================================================

def create_visualizations(df):
    """Cria visualizações dos resultados"""
    
    print("📊 GERANDO VISUALIZAÇÕES...")
    
    # Gráfico 1: Distribuição de valores de indenização
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    valores_validos = df['maior_valor'].dropna()
    if len(valores_validos) > 0:
        plt.hist(valores_validos, bins=30, alpha=0.7, color='skyblue')
        plt.title('Distribuição de Valores de Indenização')
        plt.xlabel('Valor (R$)')
        plt.ylabel('Frequência')
        plt.ticklabel_format(style='plain', axis='x')
    
    # Gráfico 2: Processos por tipo
    plt.subplot(2, 3, 2)
    consignado_counts = df['is_consignado'].value_counts()
    plt.pie(consignado_counts.values, labels=['Não Consignado', 'Consignado'], autopct='%1.1f%%')
    plt.title('Distribuição: Crédito Consignado')
    
    # Gráfico 3: Top empresas processadas
    plt.subplot(2, 3, 3)
    all_companies = [emp for lista in df['empresas'] for emp in lista]
    if all_companies:
        company_counts = Counter(all_companies).most_common(8)
        companies, counts = zip(*company_counts)
        companies_short = [c[:20] + '...' if len(c) > 20 else c for c in companies]
        plt.barh(companies_short, counts)
        plt.title('Empresas Mais Processadas')
        plt.xlabel('Número de Processos')
    
    # Gráfico 4: Valores por empresa (top 5)
    plt.subplot(2, 3, 4)
    # Criar relacionamento empresa-valor para os casos com valores
    empresa_valor = []
    for idx, row in df.iterrows():
        if not pd.isna(row['maior_valor']) and row['empresas']:
            for empresa in row['empresas']:
                empresa_valor.append({'empresa': empresa, 'valor': row['maior_valor']})
    
    if empresa_valor:
        ev_df = pd.DataFrame(empresa_valor)
        top_empresas = ev_df.groupby('empresa')['valor'].agg(['mean', 'count']).sort_values('count', ascending=False).head(5)
        plt.bar(range(len(top_empresas)), top_empresas['mean'])
        plt.xticks(range(len(top_empresas)), [e[:15] + '...' if len(e) > 15 else e for e in top_empresas.index], rotation=45)
        plt.title('Valor Médio de Indenização por Empresa')
        plt.ylabel('Valor Médio (R$)')
    
    # Gráfico 5: Confiança na classificação
    plt.subplot(2, 3, 5)
    confidence_dist = df[df['is_consignado']]['confidence'].value_counts().sort_index()
    plt.bar(confidence_dist.index, confidence_dist.values)
    plt.title('Distribuição de Confiança (Consignado)')
    plt.xlabel('Número de Padrões Matched')
    plt.ylabel('Frequência')
    
    # Gráfico 6: Valores vs Confiança
    plt.subplot(2, 3, 6)
    consignado_com_valores = df[(df['is_consignado']) & (df['qtd_valores'] > 0)]
    if len(consignado_com_valores) > 0:
        plt.scatter(consignado_com_valores['confidence'], consignado_com_valores['maior_valor'], alpha=0.6)
        plt.xlabel('Confiança (Padrões Matched)')
        plt.ylabel('Valor de Indenização (R$)')
        plt.title('Confiança vs Valor')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 9. PIPELINE COMPLETO
# =============================================================================

def run_complete_analysis(filepath):
    """Executa o pipeline completo de análise"""
    
    print("🚀 INICIANDO ANÁLISE COMPLETA")
    print("=" * 50)
    
    # 1. Carregar dados
    df = load_and_explore_data(filepath)
    
    # 2. Limpeza
    df = apply_cleaning(df)
    
    # 3. Identificar crédito consignado
    df = analyze_consignado_cases(df)
    
    # 4. Extrair valores
    df = extract_indenization_values(df)
    
    # 5. Identificar empresas
    df = analyze_companies(df)
    
    # 6. Estatísticas finais
    df = generate_statistics(df)
    
    # 7. Casos para LLM
    llm_cases = identify_llm_candidates(df)
    
    # 8. Visualizações
    create_visualizations(df)
    
    # 9. Relatório final
    print("\n🎯 RELATÓRIO FINAL")
    print("=" * 30)
    
    consignado_df = df[df['is_consignado']]
    
    if len(consignado_df) > 0:
        print(f"📊 Processos de crédito consignado analisados: {len(consignado_df)}")
        
        valores_consignado = consignado_df['maior_valor'].dropna()
        if len(valores_consignado) > 0:
            print(f"💰 Valor total em disputa: R$ {valores_consignado.sum():,.2f}")
            print(f"📈 Valor médio por processo: R$ {valores_consignado.mean():,.2f}")
        
        print(f"🤖 Casos complexos para LLM: {len(llm_cases)}")
    
    return df, llm_cases

# =============================================================================
# 10. EXEMPLOS DE USO
# =============================================================================

if __name__ == "__main__":
    # EXEMPLO DE EXECUÇÃO:
    
    # df, llm_cases = run_complete_analysis('dataset_clinica20252.csv')
    
    # TESTE INDIVIDUAL DE REGEX:
    texto_teste = """
    A requerente recebe benefício do INSS e vem sofrendo descontos 
    indevidos referentes a empréstimo consignado não contratado no 
    valor de R$ 2.700,00. Pede-se indenização por danos morais 
    no valor de 10 salários mínimos.
    """
    
    print("🧪 TESTE DE REGEX:")
    patterns = build_consignado_patterns()
    is_consig, matched = is_credito_consignado(texto_teste, patterns)
    valores = extract_monetary_values(texto_teste)
    
    print(f"É consignado: {is_consig}")
    print(f"Padrões encontrados: {matched}")
    print(f"Valores extraídos: {valores}")
