import streamlit as st
import streamlit.components.v1 as stc
# TextRank
# from gensim.summarization import summarize

# import nltk
# nltk.download('punkt')

# LexRank
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# EDA
import pandas as pd

# NLP
import spacy
nlp=spacy.load("en_core_web_sm")
from textblob import TextBlob

# styling
from spacy import displacy


# cleaning
import neattext as nt
import neattext.functions as nfx

# Data Vis
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # backend  
import seaborn as sns
import altair as alt

from wordcloud import WordCloud

# utils
from collections import Counter

def text_anal(txt):
    docx=nlp(txt)
    allData=[(token.text,token.shape_,token.pos_,token.tag_,token.lemma_,token.is_alpha,token.is_stop) for token in docx]
    allData_df=pd.DataFrame(allData,columns=['Token','Shape','PoS','Tag','Lemma','IsAlpha','Is_Stopword'])
    return allData_df


def text_ents(txt):
    docx=nlp(txt)
    entities=[(entity.text,entity.label_) for entity in docx.ents]
    return entities


HTML_WRAP="""
    <div style="overflow-x:auto;
    border: 2px solid red;
    border-radius: 30%;
    padding:1rem">
    <div class="entities" style="line-height:2.5;
    direction:ltr">
    <mark class="entity" style="background:pink;
    padding:0.45rem 0.6em;
    margin:0 0.25em;
    line-height:1;
    border-radius:0.35em;">
    text-transform:uppercase;
    vertical align:middle;
    margin-left:0.5rem">{}</span>
    </div>
"""
def style_ents(txt):
    docx=nlp(txt)
    html=displacy.render(docx,style="ent")
    html=html.replace("\n\n","\n")
    res=HTML_WRAP.format(html)
    return res

# LexRank
def sumy_sum(docx,num=2):
    parser=PlaintextParser.from_string(docx,Tokenizer("english"))
    lex_sum=LexRankSummarizer()
    sum=lex_sum(parser.document,num)
    sum_list=[str(sentence) for sentence in sum]
    res=' '.join(sum_list)
    return res

# Analyze Summary
from rouge import Rouge

def anal_sum(sum,reference):
    rg=Rouge()
    score=rg.get_scores(sum,reference)
    score_df=pd.DataFrame(score[0])
    return score_df

def common_tokens(txt,n=5):
    word_tokens=Counter(txt.split(' '))
    most_common_tokens=dict(word_tokens.most_common(n))
    return most_common_tokens

def word_cloud(txt):
    cloud=WordCloud().generate(txt)
    fig=plt.figure()
    plt.imshow(cloud,interpolation='bilinear')
    plt.axis('off')
    st.pyplot(fig)


menu=["Home","How does it work?"]
choice=st.sidebar.selectbox("Menu",menu)

if choice == "Home":
    st.title("TEXT SUMMARIZER AND ANALYZER") 
    '''
        [![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/varshney-yash) 

    '''
    st.markdown("<br>",unsafe_allow_html=True)
    # st.subheader("Something")
    raw_text=st.text_area("Paste in your text here")
    if st.button("Summarize"):
        if len(raw_text)==0:
                st.error('Hypothesis cannot be empty!')
                exit
        else:
            with st.expander("Original Text Here"):
                st.write(raw_text)
            
            with st.expander("LexRank Summary"):
                raw_text=raw_text.strip().replace('\n', ' ')	
                sum=sumy_sum(raw_text)
                l={"Original":len(raw_text),"Summary":len(sum)}
                st.write(str(l).replace("{","").replace("}", ""))
                st.write(sum)
                st.info("Rouge Score")
                score_df=anal_sum(sum,raw_text)
                # score_df=pd.DataFrame(score)
                score_df['metrics']=score_df.index
                # c=score_df.plot.bar(x="metrics", y="rouge-1", rot=70, title="Bar Chart Plot")
                score_df['metrics']=score_df.index
                c=alt.Chart(score_df).mark_bar().encode(
                    x='metrics',y='rouge-1'
                )
                st.altair_chart(c)
    
    if st.button("Analyze"):
        num_tokens=st.sidebar.number_input("Select Number of Most Common Tokens",5,20)
        with st.expander("Original Text Here"):
            st.write(raw_text)
        with st.expander("Text Analysis"):
            token_res_df=text_anal(raw_text)
            st.dataframe(token_res_df)
        with st.expander("Entities"):
            entity_res=text_ents(raw_text)
            st.write(entity_res)
            # entity_res_style=style_ents(raw_text)
            # stc.html(entity_res_style,height=1000,scrolling=True)
        c1,c2=st.columns(2)
        with c1:
            with st.expander("Word Statistics"):
                docx=nt.TextFrame(raw_text)
                st.write(str(docx.word_stats()).replace("{","").replace("}",""))
            with st.expander("Top Keywords"):
                cln_txt=nfx.remove_stopwords(raw_text)
                keywords=common_tokens(cln_txt)
                st.write(keywords)
                fig=plt.figure()
                top_keywords=common_tokens(cln_txt,num_tokens)
                plt.bar(top_keywords.keys(),top_keywords.values())
                st.pyplot(fig)

        with c2:

            with st.expander("Part of Speech Plot"):
                fig=plt.figure()
                sns.countplot(token_res_df['PoS'])
                plt.xticks(rotation=45)
                st.pyplot(fig)
            with st.expander("Word Cloud"):
                word_cloud(raw_text)
        with st.expander("Download Analysis Results (.csv)"):
                st.write("gogogo")
else: 
    st.title("Analyze Text")
    st.subheader("Theory")