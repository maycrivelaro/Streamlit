#Importando Bibliotecas
import speech_recognition as sr
import pandas as pd
import streamlit as st
from pydub import AudioSegment
from pydub.utils import make_chunks
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import time
import os
import re
import leia
from leia import SentimentIntensityAnalyzer
import datetime
import sys
from io import BytesIO
from PIL import Image
import speech_recognition as sr
import pandas as pd
import streamlit as st
from pydub import AudioSegment
from pydub.utils import make_chunks
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import time
import os
import re
import leia
from leia import SentimentIntensityAnalyzer
import datetime
import sys
from io import BytesIO
from PIL import Image

def divide(audio_wav):
    #Função para fazer a divisão do áudio importado em várias partes iguais, para otimizar o processo de transcrição

    #selecionando o áudio wav
    audio = AudioSegment.from_file(audio_wav, 'wav')

    # Tamanho em milisegundos
    tamanho = 30000

    # divisão do audio em partes
    partes = make_chunks(audio, tamanho)
    partes_audio = []
    for i, parte in enumerate(partes):
        # Enumerando arquivo particionado
        parte_name = 'teste{0}.wav'.format(i)
        # Guardando os nomes das partições em uma lista
        partes_audio.append(parte_name)
        # Exportando arquivos - salva as partes dos áudias para serem usadas na função "transcreve_audio"
        parte.export(parte_name, format='wav')
    return partes_audio

def transcreve_audio(nome_audio):
    #Função que irá transcrever parte por parte do áudio importado e depois juntar tudo em uma única transcrição

    # Selecione o audio para reconhecimento
    r = sr.Recognizer()
    with sr.AudioFile(nome_audio) as source:
        audio = r.record(source)  #leitura do arquivo de audio

    # Reconhecimento usando o Google Speech Recognition - Português Brasil
    try:
        print('Google Speech Recognition: ' + r.recognize_google(audio, language='pt-BR'))
        texto = r.recognize_google(audio, language='pt-BR')
    except sr.UnknownValueError:
        print('Google Speech Recognition NÃO ENTENDEU o audio')
        texto = ''
    except sr.RequestError as e:
        print('Erro ao solicitar resultados do serviço Google Speech Recognition; {0}'.format(e))
        texto = ''
    return texto #retorno da transcrição completa


def results(results):
    #Aqui é onde vamos montar o painel do Web APP com os resultados obtidos usando Streamlit

    ## Full Transcription
    # Expander com a transcrição completa da ligação
    st.subheader("Transcrição")
    with st.expander("Transcription"):
        text = results
        st.write(text)


    #Botão de Download em txt
    st.subheader('Download Transcrição')
    st.download_button(
        label="Download Resultados",
        data=results,
        file_name='transcrição.txt',
    )
    st.subheader(" ")

    # criar uma lista de stop_words
    stop_words = ['a', 'e', 'o', 'de', 'da', 'do', 'que']
    # criar uma wordcloud
    wc = WordCloud(stopwords=stop_words, background_color="black", width=1600, height=800)
    wordcloud = wc.generate(results)

    st.subheader("Nuvem de palavras")
    # plotar wordcloud
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_axis_off()

    # Plotar o gráfico no nosso Data App
    st.pyplot(fig)

    # Salvar o gráfico em imagem primeiro e depois abrir essa imagem de novo
    fig.savefig('wordcloud.png', format='png')
    figura = Image.open('wordcloud.png')
    buf = BytesIO()
    figura.save(buf, format="png")
    byte_im = buf.getvalue()

    #Botão de download da wordcloud no web app
    st.download_button(
        label="Download Wordcloud",
        data=byte_im,
        file_name="imagem.png",
        mime="image/png",
    )

    # Análise de sentimentos usando a biblioteca VADER adaptada ao português
    # Fonte: https://github.com/rafjaa/LeIA
    # Salvar o script "leia" na mesma venv do script do speech
    st.subheader(" ")
    st.subheader("Análise de Sentimentos")

    # Função para a análise de sentimentos
    s = SentimentIntensityAnalyzer()
    sentiment_dict = s.polarity_scores(results)

    # Configurando nosso Data App com o resultado da análise
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Positivo 😄", sentiment_dict['pos'])
    col2.metric("Negativo 😟", sentiment_dict['neg'])
    col3.metric("Neutro 😐", sentiment_dict['neu'])
    col4.metric("Composição Sentimentos", sentiment_dict['compound'])

    # Resumo e explicação das métricas utilizadas
    with st.expander("Resumo de sentimentos da ligação"):
        text = sentiment_dict
        st.write(text)
        st.write(
            "A métrica **compound** é calculada pela soma de todas as classificações normalizadas entre -1 (muito negativo) e +1 (muito positivo). E é por esta métrica que classificamos o sentimento geral da ligação!")

    st.subheader("Sentimento geral da ligação")
    if sentiment_dict['compound'] >= 0.05:
        st.success('Positivo 😄')
    elif sentiment_dict['compound'] <= -0.05:
        st.error('Negativo 😟')
    else:
        st.warning('Neutro 😐')

    sentiment_txt = str(sentiment_dict)

    #Salvando as métricas em um arquivo txt
    st.download_button(
        label="Download Análise Sentimentos",
        data=sentiment_txt ,
        file_name='analise_sentimentos.txt',
    )
    st.subheader(" ")


def main():

    #Título do web APP
    st.title("Speech to text :sunglasses:")
    #Upload do áudio em 3 formatos
    audio_file = st.file_uploader("Faça o  upload do áudio aqui", type=["mp3", "wav", "flac"])

    #Verificando se o audio_file está vazio para prosseguir com o processo
    if audio_file is not None:

        # Pegando o nome do arquivo
        name = str(audio_file.name)

        # Convertendo arquivos de áudio

        # SpeechRecognition só performa com .wav - então temos que converter qualquer outro tipo de formatao de áudio para WAV
        # Aqui aceitaremos upload apenas de mp3, flac e wav
        if name[-4:] == '.mp3':
            # Abrindo o arquivo e salvando ele em bytes
            with open(audio_file.name, 'wb') as f:
                f.write(audio_file.getbuffer())

            mp3 = name
            audio_convertido = name + '.wav'
            #Convertendo de mp3 para wav
            sound = AudioSegment.from_mp3(mp3)
            sound.export(audio_convertido, format='wav')
            st.write('O áudio foi convertido do formato mp3 para wav.')

            #Mostrando o áudio no web app
            st.audio(audio_convertido, start_time=0)

            # envio do áudio para ser particionado na função "divide"
            partes_audio = divide(audio_convertido)
            st.text("Realizando a transcrição")
            with st.spinner('Aguarde alguns instantes...'):
                time.sleep(50)

            # Transcrição das partes do áudio na função "transccreve_audio"
            # Removendo as partes de áudios que foram salvas na função "divide"

            texto = ''
            for parte in partes_audio:
                texto = texto + ' ' + transcreve_audio(parte)
                if os.path.exists(parte):
                    os.remove(parte)

            # mostrando os resultados no streamlit
            results(texto)
        elif name[-4:] == '.flac':

            # Abrindo o arquivo e salvando ele em bytes
            with open(audio_file.name, 'wb') as f:
                f.write(audio_file.getbuffer())

            flac = name
            audio_convertido = name + '.wav'
            sound = AudioSegment.from_mp3(flac)
            sound.export(audio_convertido, format='wav')
            st.write('O áudio foi convertido do formato flac para wav.')

            st.audio(audio_convertido, start_time=0)
            # envio do áudio para ser particionado
            partes_audio = divide(audio_convertido)
            st.text("Realizando a transcrição")
            with st.spinner('Aguarde alguns instantes...'):
                time.sleep(50)

            # transcrição das partes do áudio
            texto = ''
            for parte in partes_audio:
                texto = texto + ' ' + transcreve_audio(parte)
                if os.path.exists(parte):
                    os.remove(parte)

            # mostrando os resultados no streamlit
            results(texto)

        else:
            st.write('O áudio já está no formato wav.')

            st.audio(audio_file, start_time=0)
            # envio do áudio para ser particionado
            partes_audio = divide(audio_file)
            st.text("Realizando a transcrição")
            with st.spinner('Aguarde alguns instantes...'):
                time.sleep(50)

            # transcrição das partes do áudio
            texto = ''
            for parte in partes_audio:
                texto = texto + ' ' + transcreve_audio(parte)
                if os.path.exists(parte):
                    os.remove(parte)

            # mostrando os resultados no streamlit
            results(texto)


if __name__ == '__main__':
    main()