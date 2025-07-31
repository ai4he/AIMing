# --- Cell 1: Install Dependencies ---


print("Dependencies installed successfully!")
import json
import os
from openai import OpenAI # CORRECT: Import the OpenAI client
import signal
import sys
import time


MY_DEEPSEEK_API_KEY = 'sk-4a3b7fa7c48649f78ac338653b42494d'
#client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

# The model to use from the DeepSeek API
MODEL_NAME = "deepseek-coder"

# File paths
INPUT_JSON_PATH = 'output_chunks/final.json'  # Path to your large input JSON file
OUTPUT_JSON_PATH = 'defsat1/defined_chunk_2_deepseek.json' # Path to save the results
SAVE_INTERVAL = 5 # How often to save progress

# --- 1. Setup API Client ---
def setup_client():
    """Initializes the OpenAI client to connect to the DeepSeek API."""
    print("Setting up API client for DeepSeek...")
    if not MY_DEEPSEEK_API_KEY or MY_DEEPSEEK_API_KEY == "YOUR_DEEPSEEK_API_KEY":
        raise ValueError("DeepSeek API key is not set. Please set it in Colab Secrets or replace the placeholder.")

    # CORRECT: Use the OpenAI client but point it to DeepSeek's URL
    client = OpenAI(
        api_key=MY_DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com/v1"
    )
    print("Client set up successfully!")
    return client

# --- 2. Definition Extraction Function ---
def extract_definitions(latex_context, client):
    """
    Uses the DeepSeek API to extract mathematical definitions from a LaTeX string.
    """
    # The few-shot examples remain the same.
    # (Content is truncated for brevity, your full examples should be here)
    minimal_sample_latex = r"""
        \documentclass[11pt]{amsart}
      %\usepackage[dvips]{graphicx}
      \addtolength{\topmargin}{-0.7cm}
      \addtolength{\textheight}{2cm}
      \addtolength{\oddsidemargin}{-1.6cm}
      \addtolength{\evensidemargin}{-1.6cm}
      \addtolength{\textwidth}{2.7cm}
      %\usepackage{amsfonts,amsmath} \usepackage[notref,notcite]{showkeys}
      \usepackage[dvips]{graphicx} \newcommand{\halmos}{\rule{1ex}{1.4ex}}
      \newcommand{\proofbox}{\hspace*{\fill}\mbox{$\halmos$}}
      \newdimen\margin   % needed for macros \textdisplay & \ltextdisplay
      \def\COMMENT#1{}
      \def\TASK#1{}
      %\newenvironment{proof}{\noindent {\bf Proof}.}
      %{\proofbox\par\smallskip\par}

      \newenvironment{remark}{\noindent {\bf Remark}.}{\par\smallskip\par}

      \def\proof{\removelastskip\penalty55\medskip\noindent{\bf Proof. }}
      \newenvironment{proofof}[1]{\noindent {\bf Proof of
      #1}.}{\proofbox\par\smallskip\par}

      \def\noproof{{\unskip\nobreak\hfill\penalty50\hskip2em\hbox{}\nobreak\hfill%
            $\square$\parfillskip=0pt\finalhyphendemerits=0\par}\goodbreak}
      \def\endproof{\noproof\bigskip}

      \def\enddiscard{}
      \long\def\discard#1\enddiscard{}

      \newcommand{\eps}{\varepsilon}
      \newcommand{\Nat}{\mathbb{N}}
      \newcommand{\prob}{\mathbb{P}}
      \newcommand{\ex}{\mathbb{E}}
      \newcommand{\Pa}{{\mathcal P}}
      \newcommand{\G}{G_{n,p}}
      \newcommand{\la}{\ell}
      \newcommand{\ga}{\gamma}
      \newcommand{\bin}{\mathtt{Bin}}
      \newcommand{\A}{\mathtt{Aux}}
      \newcommand{\E}{\mathbb{E}}
      \newcommand{\M}{{\mathcal M}}
      \newcommand{\U}{{\mathcal U}}
      \newcommand{\B}{{\mathcal B}}
      \newcommand{\D}{{\mathcal D}}
      \newcommand{\W}{{\mathcal W}}
      \newcommand{\C}{{\mathcal C}}
      \newcommand{\cP}{{\mathcal P}}


      \newtheorem{firsttheorem}{Proposition}
      \newtheorem{fact}[firsttheorem]{Fact}
      \newtheorem{theorem}[firsttheorem]{Theorem}
      \newtheorem{thm}[firsttheorem]{Theorem}
      \newtheorem{lemma}[firsttheorem]{Lemma}
      \newtheorem{corollary}[firsttheorem]{Corollary}
      \newtheorem{conjecture}[firsttheorem]{Conjecture}
      \newtheorem{definition}[firsttheorem]{Definition}
      \newtheorem{proposition}[firsttheorem]{Proposition}
      %\newtheorem{claim}[theorem]{Claim}
      %\setlength{\oddsidemargin}{1pt}

      \begin{document}
      \title{The order of the largest complete minor in a random graph}
      \author{Nikolaos Fountoulakis, Daniela K\"uhn and Deryk
      Osthus}
      \thanks {N.~Fountoulakis and D.~K\"uhn were supported by the EPSRC, grant no.~EP/D50564X/1}
      \maketitle\vspace{-.8cm}
      \begin{abstract}
      Let~ccl($G$) denote the order of the largest complete minor in a graph~$G$ (also called
      the contraction clique number)
      and let~$G_{n,p}$ denote a random graph on~$n$ vertices with edge probability~$p$.
      Bollob\'as, Catlin and Erd\H{o}s~\cite{BCE} asymptotically
      determined~ccl($G_{n,p}$) when~$p$ is a constant.
      {\L}uczak, Pittel and Wierman~\cite{LPW} gave bounds on~ccl($G_{n,p}$)
      when~$p$ is very close to~$1/n$, i.e.~inside the phase transition.
      We show that for every $\eps>0$ there exists a constant~$C$ such that whenever
      $C/n < p <1-\eps$ then asymptotically almost surely
      ccl($G_{n,p}$)$=(1\pm \eps)n /\sqrt{\log_b (np)}$, where
      $b:=1/(1-p)$. If $p=C/n$ for a constant $C>1$, then asymptotically almost
      surely ccl($G_{n,p}$)$=\Theta(\sqrt{n})$.
      This extends the results in~\cite{BCE} and answers a question
      of Krivelevich and Sudakov~\cite{KS}.
      \end{abstract}


      \section{Introduction}
      \subsection{Main results}

      A graph~$H$ is a \emph{minor} of~$G$ if for every vertex $h\in H$ there
      is a connected subset $B_h\subseteq V(G)$ such that all the~$B_h$ are disjoint
      and~$G$ contains an edge between~$B_h$ and~$B_{h'}$ whenever~$hh'$ is an edge of~$H$.
      The~$B_h$'s are called the \emph{branch sets}.
      We denote by~ccl$(G)$ the order of the largest complete minor in~$G$.
      The study of the largest complete minor contained in
      a given graph has its origins in Hadwiger's conjecture which
      states that if the chromatic number of a graph~$G$ is at least~$k$,
      then~$G$ contains $K_k$~minor. It has been proved for $k\le 6$
      (see for example~\cite[Chapter~7]{Diest} for a discussion).

      Bollob\'as, Catlin and Erd\H{o}s~\cite{BCE} proved that Hadwiger's conjecture is
      true for almost all graphs. For this, they
      estimated the typical order of the largest complete minor in
      a graph on~$n$ vertices and compared it with the typical chromatic number of such a graph.
      In particular, they proved that for constant~$p$ and $\eps >0$
      asymptotically almost surely ccl$(G_{n,p})=(1 \pm \eps)n/\sqrt{\log_b n}$,
      where $b:=1/(1-p)$. Here~$G_{n,p}$ is a random graph on~$n$ vertices
      where the edges are present independently and with probability~$p$.
      We say that an event occurs \emph{asymptotically almost surely} (a.a.s.)
      if it occurs with probability tending to~$1$ as~$n$ tends to infinity.

      Krivelevich and Sudakov~\cite{KS} considered the order of the largest
      complete minor in a sparser random graph (and more generally in arbitrary
      pseudo-random and expanding graphs).
      %They observed that the proof in~\cite{BCE} can be extended to the case
      %$p \to 0$ as long as $p$ is not too small, but that it breaks down eventually.
      They determined the order of magnitude of~ccl$(G_{n,p})$
      as long as~$p\ge n^{\eps-1}$. Our first result determines~ccl$(G_{n,p})$ asymptotically
      as long as $p\ge C/n$ and $p=o(1)$.

      \begin{thm}\label{thmdense}
      For every $\eps>0$ there exists a constant $C=C(\eps)$ such that if $pn\ge C$
      and $p=o(1)$, then a.a.s.
      $${\rm ccl}(G_{n,p})=(1\pm \eps) \sqrt{\frac{n^2p}{\ln (np)}}. $$
      \end{thm}

      One can combine Theorem~\ref{thmdense} with~\cite{BCE} to obtain
      a single formula which allows for constant~$p$ as well. Indeed, let $b:=1/(1-p)$.
      If $p=o(1)$ a series expansion gives $\ln b=-\ln (1-p)=p+O(p^2)$.
      Thus
      $$
      \sqrt{\frac{n^2p}{\ln (np)}} = \sqrt{\frac{n^2 p}{\ln b\log_b (np)}}
      =(1+o(1)){n \over \sqrt{\log_b (np)}}.
      $$
      Also if~$p$ is constant, then $\log_b n=(1+o(1))\log_b(np)$.
      So altogether we obtain the following.
      \begin{corollary}\label{thmdense1}
      For every  $\eps>0$ there exists a constant $C=C(\eps)$ such that if $C/n \le p \le 1-\eps$,
      then a.a.s.
      $${\rm ccl}(G_{n,p})=(1\pm \eps) \frac{n}{\sqrt{\log_{b} (np)}}. $$
      \end{corollary}

      In the last section of the paper, we estimate~ccl$(G_{n,c/n})$ where $c>1$ is fixed.
      Krivelevich and Sudakov~\cite{KS} observed that there are constants~$c_1$ and~$c_2$
      such that $c_1\sqrt{n /\log n}\le {\rm ccl}(G_{n,c/n})\le c_2\sqrt{n}$
      and asked what the correct order of magnitude is.

      """

    minimal_sample_3 = r"""\documentclass[a4paper,11pt]{article}
    \usepackage{enumerate}
    \usepackage{latexsym}
    \usepackage{amsfonts}
    \usepackage[only,ninrm,elvrm,twlrm,sixrm,egtrm,tenrm]{rawfonts}
    \usepackage{indentfirst}
    \usepackage{amsmath}
    \usepackage[noend]{algorithmic}
    \usepackage{algorithm}
    \usepackage{graphicx,psfrag}
    \usepackage{graphics}
    \usepackage{makeidx}
    \parindent 15pt
    \newtheorem{thm}{Theorem}[section]
    \newtheorem{cor}[thm]{Corollary}
    \newtheorem{lem}[thm]{Lemma}
    \newtheorem{op}[thm]{Open Problem}
    \newtheorem{exam}{Example}
    \newtheorem{prop}[thm]{Proposition}
    \newtheorem{defn}[thm]{Definition}
    \newtheorem{rem}{Remark}
    \baselineskip=15pt
    \def\qed{\hfill \nopagebreak\rule{5pt}{8pt}}
    \title{\bf Partitioning complete graphs by heterochromatic trees
     \footnote{Supported by NSFC, PCSIRT and the ``973" program. }}

    \author{
    \small  Zemin Jin$^1$ and Xueliang Li$^2$ \\
    [3mm] \small    $^{1}$Department of Mathematics, Zhejiang Normal University\\
    \small Jinhua 321004, P.R. China\\
    \small $^{2}$Center for Combinatorics and LPMC,  Nankai University\\
    \small Tianjin 300071, P.R. China\\}

    \begin{document}

    \maketitle
    \begin{abstract}

    A {\it heterochromatic tree} is an edge-colored tree in which any
    two edges have different colors. The {\it heterochromatic tree
    partition number} of an $r$-edge-colored graph $G$, denoted by
    $t_r(G)$, is the minimum positive integer $p$ such that whenever
    the edges of the graph $G$ are colored with $r$ colors, the
    vertices of $G$ can be covered by at most $p$ vertex-disjoint
    heterochromatic trees. In this paper we determine the
    heterochromatic tree partition number of an $r$-edge-colored
    complete graph.\\
    [0.1in] {\bf  Keywords:}  edge-colored graph, heterochromatic
    tree, partition.\\
    [2mm] {\bf AMS subject classification (2000)}: 05C05, 05C15, 05C70,
    05C75.
    \end{abstract}

    \section{Introduction}
    A {\it monochromatic (heterochromatic) tree} is an edge-colored
    tree in which any two edges have the same (different) color(s).
    The {\it (monochromatic) tree partition number} of an
    $r$-edge-colored graph $G$ is defined to be the minimum positive
    integer $p$ such that whenever the edges of $G$ are colored with
    $r$ colors, the vertices of $G$ can be covered by at most $p$
    vertex-disjoint monochromatic trees. The {\it (monochromatic)
    cycle partition number} and the {\it (monochromatic)
    path partition number} are defined similarly.\\

    Erd\H{o}s, Gy\'{a}rf\'{a}s and Pyber \cite{egp} proved that the
    (monochromatic) cycle partition number of an $r$-edge-colored
    complete graph $K_n$ is at most $cr^2\ln r $ for some constant
    $c$. This implies a conjecture from \cite{gys} in a stronger form.
    Recently, the bound was improved by Gy\'{a}rf\'{a}s et al.
    \cite{gyfs}. Almost solving one of the conjectures in \cite{egp},
    Haxell and Kohayakawa \cite{hk} proved that the (monochromatic)
    tree partition number of an $r$-edge-colored complete graph $K_n$
    is at most $r$ provided that $n$ is large enough with respect to
    $r$. Haxell \cite{ha} proved that the (monochromatic) cycle
    partition number of an $r$-edge-colored complete bipartite graph
    $K_{n,n}$ is also independent of $n$, which answered a question in \cite{egp}.\\

    From above, one can see that the (monochromatic) tree, path, and
    cycle partition number of $r$-edge-colored graphs $K_{n}$ and
    $K_{n,n}$ are independent of $n$. The same seems to be not true
    for other graphs. Also, no (monochromatic) partition number of an
    $r$-edge-colored graph $K_{n}$ or $K_{n,n}$ is determined exactly.
    The only exception is due to Kaneko, Kano and Suzuki \cite{kks},
    who gave an explicit expression for the (monochromatic) tree
    partition number of a 2-edge-colored complete multipartite graph.
    In particular, let $n_1, n_2, \cdots, n_k$ ($k\geq 2$) be integers
    such that $1\leq n_1\leq n_2\leq \cdots \leq n_k$ and let $n=n_1+
    n_2+\cdots +n_{k-1}, m=n_k$. The authors \cite{kks} proved that
    $$
    t_2^{'}(K_{n_1, n_2, \cdots, n_k})=\lfloor \frac{m-2}{2^n}
    \rfloor+2,
    $$
    where $t_r^{'}(K_{n_1, n_2, \cdots, n_k})$ denotes the
    (monochromatic) tree partition number of the $r$-edge-colored graph
    $K_{n_1, n_2, \cdots, n_k}$. Other related partition
    problems can be found in \cite{eg,Luczak,rado}.\\

    Analogous to the monochromatic tree partition case, the authors
    \cite{chen} introduced the definition of {\it heterochromatic tree
    partition number} of an $r$-edge-colored graph $G$. The {\it
    heterochromatic tree partition number} of an $r$-edge-colored
    graph $G$, denoted by $t_r(G)$, is defined to be the minimum
    positive integer $p$ such that whenever the edges of the graph $G$
    are colored with $r$ colors, the vertices of $G$ can be covered by
    at most $p$ vertex-disjoint heterochromatic trees. In \cite{chen},
    the authors determined the heterochromatic tree partition number
    of an $r$-edge-colored complete bipartite graph $K_{m,n}$. In this
    paper we consider an $r$-edge-colored complete graph $K_{n}$ and give
    the exact expression for its heterochromatic tree partition number. \\

    Before proceeding, we introduce some definitions and notations.
    Throughout this paper, we use $r$ to denote the number of the
    colors, and an $r$-edge-coloring of a graph $G$ means that each
    color appears at least once in $G$. Let $\phi$ be an
    $r$-edge-coloring of a graph $G$. For an edge $e\in E(G)$, denote
    by $\phi(e)$ the color of $e$. Denote by $t_r(G, \phi)$ the
    minimum positive integer $p$ such that under the $r$-edge-coloring
    $\phi$, the vertices of $G$ can be covered by at most $p$
    vertex-disjoint heterochromatic trees. Clearly, $t_r(G)=\max
    _{\phi} t_r(G, \phi)$, where $\phi$ runs over all
    $r$-edge-colorings of the graph $G$. Let $\phi$ be an
    $r$-edge-coloring of the graph $G$ and $F$ be a spanning forest of
    $G$, each component of which is a heterochromatic tree. If $F$
    contains exactly $t_r(G, \phi)$ components, then $F$ is called an
    {\it optimal heterochromatic tree partition} of the graph $G$ with
    edge-coloring $\phi$. Note that a tree consisting of a single
    vertex is also regarded as a heterochromatic tree. \\

    For any integer $r\geq 2$, there is a unique positive integer $t$,
    such that ${t\choose 2} +2\leq r\leq {{t+1}\choose 2} +1$.
    Clearly, the integer $t$ is determined completely by $r$, and here
    we denote it by $f(r)=t$. This integer $f(r)=t$ will play an
    important role in expressing the number $t_r(K_n)$. If the color
    number $r=1$, clearly a maximum matching (plus a single vertex
    when $n$ is odd) in $K_n$ is an optimal heterochromatic tree
    partition, and then $t_r(K_n)=\lceil \frac{n}{2} \rceil$. So, in
    the rest of this paper we only consider the case $2\leq r\leq
    {n\choose 2}$. The following is the main result of this paper.

     """

    example_definition_generation = r"""\subsection*{Definitions}
    \begin{enumerate}
        \item \textbf{Random Graph $G(n,p)$:} A graph on $n$ labeled vertices where each of the $\binom{n}{2}$ possible edges is present independently with probability $p$.
        \item \textbf{Graph Minor:} A graph $H$ is a \emph{minor} of $G$ if for every vertex $h \in V(H)$ there is a connected subset $B_h \subseteq V(G)$ such that all the $B_h$ are disjoint and $G$ contains an edge between $B_h$ and $B_{h'}$ whenever $hh'$ is an edge of $H$.
        \item \textbf{Contraction Clique Number, $\mathrm{ccl}(G)$:} The largest integer $k$ such that $K_k$ is a minor of $G$.
        \item \textbf{Asymptotically Almost Surely (a.a.s.):} An event occurs with probability tending to 1 as the number of vertices $n$ tends to infinity.
    \end{enumerate}
    """

    example_definition_generation_3 = r"""\subsection*{Definitions}
    \begin{enumerate}
        \item \textbf{$r$-edge-coloring:} An assignment of $r$ distinct colors to the edges of a graph $G$. It is assumed every color is used at least once.
        \item \textbf{Heterochromatic Tree:} A \emph{heterochromatic tree} is an edge-colored tree in which any two edges have different colors.
        \item \textbf{Heterochromatic Tree Partition:} A partition of the vertex set $V(G)$ into a collection of vertex-disjoint heterochromatic trees.
        \item \textbf{Heterochromatic Tree Partition Number, $t_r(G)$:} The \emph{heterochromatic tree partition number} of an $r$-edge-colored graph $G$, denoted by $t_r(G)$, is the minimum positive integer $p$ such that whenever the edges of the graph $G$ are colored with $r$ colors, the vertices of $G$ can be covered by at most $p$ vertex-disjoint heterochromatic trees.
    \end{enumerate}"""

    # Construct the messages payload for the API
    messages = [
        {
            "role": "system",
            "content": "You are an expert AI assistant that reads mathematical papers in LaTeX. Your sole task is to extract key definitions and present them as a clear, formatted list. Do not add any extra commentary."
        },
        {
            "role": "user", "content": f"Create a list of all mathematical terms defined in the following text.\n\n**Input Text:**\n{minimal_sample_latex}"
        },
        {
            "role": "assistant", "content": example_definition_generation
        },
        {
            "role": "user", "content": f"Excellent. Now do the same for this new text.\n\n**Input Text:**\n{minimal_sample_3}"
        },
        {
            "role": "assistant", "content": example_definition_generation_3
        },
        {
            "role": "user", "content": f"Excellent. Now do the same for this new text.\n\n**Input Text:**\n{latex_context}"
        }
    ]

    # Make the API call using the client
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=1024,
        temperature=0.1, # Lower temperature for more consistent, deterministic output
    )

    # Extract and return the result
    result = response.choices[0].message.content
    return result

# --- 3. Main Processing Logic ---
def main():
    """Main function to run the processing pipeline."""
    client = setup_client()

    # Load the full dataset (make sure Google Drive is mounted)
    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            all_papers = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{INPUT_JSON_PATH}'")
        print("Please ensure your Google Drive is mounted (`from google.colab import drive; drive.mount('/content/drive')`) and the file path is correct.")
        return

    # Load already processed data or initialize an empty list
    if os.path.exists(OUTPUT_JSON_PATH):
        with open(OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            try:
                processed_papers = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Output file '{OUTPUT_JSON_PATH}' is corrupted or empty. Starting fresh.")
                processed_papers = []
    else:
        processed_papers = []

    processed_ids = {p.get('id') for p in processed_papers}
    print(f"Found {len(processed_ids)} already processed papers.")

    # Graceful shutdown handler
    def save_and_exit(sig, frame):
        print("\nSignal received. Saving progress before exiting...")
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(processed_papers, f, indent=4, ensure_ascii=False)
        print(f"Progress saved to '{OUTPUT_JSON_PATH}'. Exiting.")
        sys.exit(0)

    signal.signal(signal.SIGINT, save_and_exit)

    # Process all papers, skipping those already done
    for i, paper in enumerate(all_papers):
        paper_id = paper.get('id')
        if paper_id in processed_ids:
            continue

        print(f"[{i+1}/{len(all_papers)}] Processing paper ID: {paper_id}")

        try:
            if 'previous context' not in paper or not paper['previous context']:
                 raise ValueError("'previous context' field is missing or empty.")

            definitions_text = extract_definitions(paper['previous context'], client)
            new_entry = paper.copy()
            new_entry['definitions'] = definitions_text
            processed_papers.append(new_entry)
            time.sleep(1) # Add a small delay to respect API rate limits

        except Exception as e:
            print(f"--- ERROR processing {paper_id}: {e} ---")
            error_entry = paper.copy()
            error_entry['definitions'] = f"PROCESSING_ERROR: {str(e)}"
            processed_papers.append(error_entry)

        # Save progress periodically
        if (i + 1) % SAVE_INTERVAL == 0:
            print(f"Saving progress to '{OUTPUT_JSON_PATH}'...")
            with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                json.dump(processed_papers, f, indent=4, ensure_ascii=False)

    # Final save
    print("Processing complete. Performing final save...")
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(processed_papers, f, indent=4, ensure_ascii=False)
    print(f"All done. Final results saved to '{OUTPUT_JSON_PATH}'.")

main()