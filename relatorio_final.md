# Relatório final — classificação de tópicos

## a) Sistema

O comando `python run_final.py` executa uma pipeline offline:

1. carrega as 659 questões, os 377 temas e a folha-verdade;
2. usa somente temas ativos e constrói a relação pai/filho da árvore;
3. usa TF-IDF de palavras (unigramas/bigramas) e de caracteres (3–5);
4. usa similaridade dos embeddings BGE-M3 versionados em `artefatos/`;
5. combina os três sinais com pesos ajustados nos rótulos;
6. agrega a pontuação pela árvore e escolhe o tema central;
7. escolhe o threshold pelo maior prefixo OOF com precisão mínima de 96%;
8. publica, revisa ou envia para fora.

O corpus `eduagarcia/oab_exams` é usado somente para ampliar o corpus lexical
e ajustar o IDF. São usados enunciados e alternativas não anuladas; `answerKey`,
respostas e rótulos externos são ignorados. A decisão final não chama LLM.

Foram testados e descartados: regressão logística, SVM linear e Ridge como
classificadores principais; todos ficaram perto de 38–39% de top-1 OOF devido
à relação entre 100 rótulos e centenas de temas. Também testamos perfis com
somente 10, 20 e 30 termos discriminativos. Houve ganho marginal de precisão,
mas perda de cobertura; por isso mantivemos os perfis completos.

## b) Os 100 rótulos

Os 100 registros foram preservados e usados como exemplos supervisionados para
ajustar os pesos e como base da validação cruzada OOF. O threshold não é
escolhido avaliando as 559 questões sem rótulo: cada questão rotulada é
predita em folds nos quais ela não participa do ajuste. As cinco repetições
com dez folds produzem a distribuição de confiança usada na política.

A folha também define a convenção de avaliação: publicar no tema central, em
um extra ou em um pai/filho do mesmo ramo é acerto; publicar em outro irmão é
erro. A validação mostrou top-1 OOF de aproximadamente 79,8% e precisão de
96,02% no subconjunto que a política teria publicado. Isso é uma estimativa
OOF, não uma prova independente da precisão populacional das 559 questões.

## c) Diagnóstico

Os erros mais prováveis são:

- temas irmãos com vocabulário semelhante;
- temas genéricos que atraem questões de várias matérias;
- assunto de origem incorreto ou amplo;
- enunciados que descrevem um crime, mas cobram outro instituto jurídico;
- questões com pouca evidência lexical para um subtema específico.

O sistema encaminha casos abaixo do threshold para revisão para evitar
transformar incerteza em publicação. As dez questões `fora` são um gate de
escopo para disciplina de origem militar, que não pertence à taxonomia deste
recorte. As melhorias prioritárias são revisar nomes/descrições de temas
genéricos, separar irmãos ambíguos e registrar extras quando a questão cobre
mais de um tema.

## d) Produção

Na máquina usada, a execução final reproduzível levou cerca de 9,5 segundos
com embeddings em cache e o parquet da OAB disponível localmente. Para 50 mil
questões semanais, a etapa lexical e as multiplicações esparsas podem ser
processadas em lotes e paralelizadas em CPU. O custo maior é gerar embeddings:
eles devem ser produzidos em worker GPU, armazenados em cache e reutilizados.
Como estimativa inicial, reservaria uma janela de minutos a dezenas de minutos
por lote, dimensionando por benchmark do encoder e não pelo classificador.

Cada correção do professor deve registrar questão, tema central, extras,
motivo e versão da árvore. As correções aprovadas entram no conjunto de treino;
o pipeline deve ser retreinado semanalmente ou ao acumular uma quantidade
relevante de correções. Threshold, árvore e qualidade devem ser revisados
mensalmente, ou imediatamente se uma amostra auditada cair abaixo de 95%.

## e) Memo para o professor

O sistema deve publicar apenas questões com evidência forte. Questões com
temas irmãos próximos ou baixa margem ficam em revisão. Os principais pedidos
ao professor são: confirmar quais temas genéricos devem ser divididos,
identificar nomes que são sinônimos, marcar questões que cobrem mais de um
tema e corrigir os casos de assunto de origem errado. Essas correções viram
exemplos rastreáveis para o próximo retreinamento.

## Entrega e limitações

`publicacao.csv` contém 659 decisões; `previsao.txt` informa a previsão para
as 559 fora da folha. A confiança é um score operacional, não uma probabilidade
populacional comprovada. O único comando oficial de regeneração é
`python run_final.py`; os artefatos em `artefatos/` fazem parte da solução.
