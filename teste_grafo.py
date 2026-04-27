import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ===== DEFINIÇÕES PARA O GRAFO DE DATASETS E TÉCNICAS DE IMPUTAÇÃO =====

# 1. Criar o grafo bipartido
G2 = nx.Graph()

# 2. Definir os datasets (lado esquerdo do grafo) - NOMES SIMPLIFICADOS
datasets = [
    'Saúde',
    'Tráfego',
    'Saúde \n Ambiental',
    'Repositório',
    'Energia',
    'Clima\nTemperatura',
    'Geofísica',
    'Sensores Industriais',
    'Detecção de Anomalias',
    'Segurança\nCibernética',
    'MNIST',
    'C-MAPSS'
]

# 3. Definir as técnicas de imputação (lado direito do grafo)
tecnicas = [
    'RNN',
    'GNN',
    'Transformers',
    'GAN',
    'Métodos Estatísticos',
    'Método \n Gaussiano',
    'DAE',
    'SAITS',
    'VAE',
    'MAE',
    'MTSC'
]

# 4. Adicionar os nós ao grafo
G2.add_nodes_from(datasets, bipartite=0)
G2.add_nodes_from(tecnicas, bipartite=1)

# 5. Adicionar arestas com pesos (conexões entre datasets e técnicas)
conexoes = [
    # Saúde
    ('Saúde', 'RNN', 1),
    ('Saúde', 'GNN', 1),
    ('Saúde', 'Transformers', 2),
    ('Saúde', 'GAN', 2),
    ('Saúde', 'Métodos Estatísticos', 1),
    ('Saúde', 'Método \n Gaussiano', 1),
    ('Saúde', 'DAE', 1),
    
    # Tráfego
    ('Tráfego', 'RNN', 2),
    ('Tráfego', 'GNN', 5),
    ('Tráfego', 'SAITS', 1),
    ('Tráfego', 'Métodos Estatísticos', 1),
    ('Tráfego', 'VAE', 1),
    
    # Saúde Ambiental
    ('Saúde \n Ambiental', 'RNN', 1),
    ('Saúde \n Ambiental', 'GNN', 1),
    ('Saúde \n Ambiental', 'Transformers', 1),
    
    # Repositório
    ('Repositório', 'RNN', 1),
    ('Repositório', 'GNN', 1),
    ('Repositório', 'Transformers', 2),
    ('Repositório', 'SAITS', 1),
    ('Repositório', 'Métodos Estatísticos', 1),
    ('Repositório', 'Método \n Gaussiano', 2),
    
    # Energia
    ('Energia', 'RNN', 1),
    ('Energia', 'Transformers', 1),
    ('Energia', 'SAITS', 1),
    ('Energia', 'Métodos Estatísticos', 1),
    ('Energia', 'Método \n Gaussiano', 1),
    ('Energia', 'MAE', 1),
    ('Energia', 'MTSC', 1),
    
    # Clima/Temperatura
    ('Clima\nTemperatura', 'GNN', 1),
    ('Clima\nTemperatura', 'SAITS', 1),
    ('Clima\nTemperatura', 'Métodos Estatísticos', 1),
    ('Clima\nTemperatura', 'MAE', 1),
    
    # Geofísica
    ('Geofísica', 'RNN', 1),
    ('Geofísica', 'GNN', 1),
    
    # Sensores Industriais
    ('Sensores Industriais', 'RNN', 1),
    ('Sensores Industriais', 'Transformers', 1),
    
    # Detecção de Anomalias
    ('Detecção de Anomalias', 'RNN', 1),
    ('Detecção de Anomalias', 'Transformers', 1),
    ('Detecção de Anomalias', 'VAE', 1),
    
    # Segurança Cibernética
    ('Segurança\nCibernética', 'RNN', 1),
    ('Segurança\nCibernética', 'VAE', 1),
    
    # MNIST
    ('MNIST', 'GNN', 1),
    ('MNIST', 'SAITS', 1),
    
    # C-MAPSS
    ('C-MAPSS', 'RNN', 1),
]

for dataset, tecnica, peso in conexoes:
    G2.add_edge(dataset, tecnica, weight=peso)

# 6. Definir a ordem dos nós
cat_order = datasets
tec_order = tecnicas

# ===== CÓDIGO DE VISUALIZAÇÃO =====

def wrap_label(label_txt, max_line_len):
    parts = label_txt.split(' ')
    lines = []
    cur = ''
    for w in parts:
        cand = (cur + ' ' + w).strip() if cur else w
        if len(cand) <= max_line_len:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    if len(lines) > 2:
        lines = lines[:2]
        if len(lines[1]) > 3:
            lines[1] = lines[1][:-3] + '...'
    return '\n'.join(lines)

# Wrapped labels (2 lines max) to reduce needed circle size and height
wrapped_labels = {}
for n in G2.nodes():
    if len(n) > 18:
        wrapped_labels[n] = wrap_label(n, 16)
    else:
        wrapped_labels[n] = n

# TAMANHO DOS CÍRCULOS - REDUZIDOS DRASTICAMENTE
base_size5 = 4000        # Reduzido de 700 para 400
size_scale5 = 0        # Reduzido de 35 para 15 (crescimento mínimo)

def adjusted_node_size(label_txt):
    ln = len(label_txt)
    size_val = base_size5 + size_scale5 * (ln ** 2)
    # Aplicar redução mais agressiva para textos longos
    if ln >= 22:
        size_val = size_val * 1.00    # Reduzido de 0.58 para 0.45
    elif ln >= 18:
        size_val = size_val * 1.00    # Reduzido de 0.72 para 0.60
    return float(size_val)

cat_sizes5 = [adjusted_node_size(n) for n in cat_order]
tec_sizes5 = [adjusted_node_size(n) for n in tec_order]

# Size-aware spacing
radius_pts_cat5 = [np.sqrt(s / np.pi) for s in cat_sizes5]
radius_pts_tec5 = [np.sqrt(s / np.pi) for s in tec_sizes5]

# ESPAÇAMENTO - mantendo o valor alto que você definiu
base_gap_pts5 = 60     # Mantido em 200

def cumulative_y_from_radii(radius_pts_list, gap_pts):
    y_vals = [0.0]
    for k in range(1, len(radius_pts_list)):
        step = radius_pts_list[k-1] + radius_pts_list[k] + gap_pts
        y_vals.append(y_vals[-1] - step)
    return y_vals

cat_y_pts5 = cumulative_y_from_radii(radius_pts_cat5, base_gap_pts5)
tec_y_pts5 = cumulative_y_from_radii(radius_pts_tec5, base_gap_pts5)

# Fator de conversão vertical
pts_to_axis5 = 0.080
cat_y5 = [v * pts_to_axis5 for v in cat_y_pts5]
tec_y5 = [v * pts_to_axis5 for v in tec_y_pts5]

pos5 = {}
for i, n in enumerate(cat_order):
    pos5[n] = (0, cat_y5[i])
for j, n in enumerate(tec_order):
    pos5[n] = (1, tec_y5[j])

# Edge styling
weights5 = [G2[u][v]['weight'] for u, v in G2.edges()]
max_w5 = max(weights5) if len(weights5) else 1
edge_widths5 = [0.8 + 3.0 * (w / max_w5) for w in weights5]

# Figure height
all_y5 = cat_y5 + tec_y5
y_range5 = (max(all_y5) - min(all_y5)) if len(all_y5) else 10
fig_h5 = max(10, min(35, y_range5 * 2.5))  # Aumentei o limite máximo para 28

plt.figure(figsize=(20, fig_h5))

# Datasets em azul
nx.draw_networkx_nodes(G2, pos5, nodelist=cat_order, 
                       node_color='#1f77b4',
                       node_size=cat_sizes5, 
                       alpha=0.97, 
                       linewidths=1.2, 
                       edgecolors='#0d3d5c')

# Técnicas em laranja suave
nx.draw_networkx_nodes(G2, pos5, nodelist=tec_order, 
                       node_color='#ff9966',
                       node_size=tec_sizes5, 
                       alpha=0.97, 
                       linewidths=1.2, 
                       edgecolors='#cc6633')

# Linhas em vermelho escuro
nx.draw_networkx_edges(G2, pos5, 
                       width=edge_widths5, 
                       edge_color='#8B0000',
                       alpha=0.40,
                       connectionstyle='arc3,rad=0.15')

nx.draw_networkx_labels(G2, pos5, labels=wrapped_labels, font_size=10, font_color='black')

plt.title('Grafo Bipartido: Datasets × Técnicas de Imputação', fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.show()

# Salvar o grafo
out_png5 = 'grafo_bipartido_datasets_tecnicas_imputacao.png'
plt.figure(figsize=(20, fig_h5))

nx.draw_networkx_nodes(G2, pos5, nodelist=cat_order, 
                       node_color='#1f77b4', 
                       node_size=cat_sizes5, 
                       alpha=0.97, 
                       linewidths=1.2, 
                       edgecolors='#0d3d5c')

nx.draw_networkx_nodes(G2, pos5, nodelist=tec_order, 
                       node_color='#ff9966', 
                       node_size=tec_sizes5, 
                       alpha=0.97, 
                       linewidths=1.2, 
                       edgecolors='#cc6633')

nx.draw_networkx_edges(G2, pos5, 
                       width=edge_widths5, 
                       edge_color='#8B0000',
                       alpha=0.40, 
                       connectionstyle='arc3,rad=0.15')

nx.draw_networkx_labels(G2, pos5, labels=wrapped_labels, font_size=10, font_color='black')

plt.title('Grafo Bipartido: Datasets × Técnicas de Imputação', fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig(out_png5, dpi=300, bbox_inches='tight')
plt.show()

print(f'✅ Grafo salvo em: {out_png5}')
print(f'\n📊 Estatísticas do grafo:')
print(f'   • Total de datasets: {len(datasets)}')
print(f'   • Total de técnicas: {len(tecnicas)}')
print(f'   • Total de conexões: {len(conexoes)}')
print(f'\n🎨 Configurações aplicadas:')
print(f'   • Círculos MUITO menores (base: 400, escala: 15)')
print(f'   • Espaçamento: {base_gap_pts5} pontos')
print(f'   • Redução agressiva para textos longos')
print(f'   • Linhas vermelho escuro')