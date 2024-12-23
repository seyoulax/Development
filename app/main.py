import numpy as np
import streamlit as st
import pandas as pd
from catboost_predictor import get_predict
import plotly.express as px
import glob
from catboost_predictor import inference_worker
import gdown

if "test_X.csv" not in glob.glob("*"):
    gdown.download("https://drive.google.com/uc?id=1T0s79KHWpjLRBZ9uMU4bFHV8WLzbX5yd")

if "data" not in st.session_state:
    st.session_state.data = pd.read_csv("test_X.csv")
    preds = get_predict(st.session_state.data)
    st.session_state.data["default6"] = preds

if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'expanded'
st.set_page_config(initial_sidebar_state=st.session_state.sidebar_state,
                   layout="wide")


st.sidebar.image("logo.png", width=240)
st.sidebar.write("Интеллектуальные предсказания для вашего бизнеса")

selecter_bar = st.sidebar.radio("", ["Загрузить данные", "BI аналатика"])

if selecter_bar == "Загрузить данные":
    st.markdown("### Для аналитика загрузите данные")
    st.write("Стандартно в систему загружена тестовая выборка")
    data_option = st.selectbox("Выберите набор данных", ("Стандартный", "Загрузить"))
    col1, col2 = st.columns((5, 5))
    col21, col22 = col2.columns((3, 6))
    if data_option == "Загрузить":
        uploaded_file = st.file_uploader("Выберите файл", type=["csv"])
        if uploaded_file is not None:
            dataframe = pd.read_csv(uploaded_file)
            col1.write(dataframe)

            preds = get_predict(dataframe)
            dataframe["default6"] = preds
            st.session_state.data = dataframe
            download_button = col1.download_button("Скачать данные с предсказаниями",
                                                   st.session_state.data.to_csv(index=False), "data.csv")
            tmp = st.session_state.data.drop(["contract_date", "report_date"], axis=1).groupby("contract_id").agg(np.max)
            brokend_ids = tmp[tmp["default6"] > 0.5].index
            col2.write(f"Обнаружно {len(brokend_ids)} контрактов, которые могут сорваться в ближайшее время")
            col21.write(brokend_ids)
            fig = px.histogram(tmp["default6"], x=tmp["default6"], nbins=100, title="Распределение вероятностей срыва\n внутри контрактов")
            col22.plotly_chart(fig, use_container_width=True)
            col2.markdown("**Вы можете узнать больше информации в BI дашборде**")


    else:
        dataframe = pd.read_csv("test_X.csv")
        col1.write(dataframe)
        preds = get_predict(dataframe)
        dataframe["default6"] = preds
        st.session_state.data = dataframe
        st.write("Стандартная тестовая выборка успешно загружена ✅")
        download_button = col1.download_button("Скачать данные с предсказаниями",
                                               st.session_state.data.to_csv(index=False), "data.csv")
        tmp = st.session_state.data.drop(["contract_date", "report_date"], axis=1).groupby("contract_id").agg(np.max)
        brokend_ids = tmp[tmp["default6"] > 0.5].index
        col2.write(f"Обнаружно {len(brokend_ids)} контрактов, которые могут сорваться в ближайшее время")
        col21.write(brokend_ids)
        fig = px.histogram(tmp["default6"], x=tmp["default6"], nbins=100, title="Распределение вероятностей срыва\n внутри контрактов")
        col22.plotly_chart(fig, use_container_width=True)
        col2.markdown("**Вы можете узнать больше информации в BI дашборде**")

elif selecter_bar == "BI аналатика":
    st.markdown("### BI аналитика")
    col1, col2 = st.columns((7, 3))
    use0 = col2.checkbox("Рискованные", value=True)
    use1 = col2.checkbox("Безопасные", value=False)
    contrs = []
    tmp = st.session_state.data.drop(["contract_date", "report_date"], axis=1).groupby("contract_id").agg(np.max)
    if use0:
        contrs.extend(tmp[tmp["default6"] > 0.5].index)
    if use1:
        contrs.extend(tmp[tmp["default6"] < 0.5].index)

    selected_id = col1.selectbox("Выберите контракт", contrs)

    df = st.session_state.data[st.session_state.data["contract_id"] == selected_id]

    col1, col2 = st.columns((5, 5))
    col1.write(df)
    fig = px.line(x=df["report_date"], y=df["default6"],
                       title="Вероятности срыва\n внутри контрактов")
    col2.plotly_chart(fig, use_container_width=True)

    col1.write("Распределение значимости признаков")
    col1.write(inference_worker.folds_models[0].get_feature_importance(prettified=True))