import streamlit as st
import streamlit_antd_components as sac
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_searchbox import st_searchbox
import os
import gdown

if not os.path.exists("df.csv"):
    gdown.download("https://drive.google.com/uc?id=1qNtkFhaOUVfvbFXWqxzVgwQEs3IEn9pt")

pd.set_option("styler.render.max_elements", 1030848)


if "sidebar_state" not in st.session_state:
    st.session_state.sidebar_state = "expanded"
st.set_page_config(initial_sidebar_state=st.session_state.sidebar_state, layout="wide")

if "df" not in st.session_state:
    df = pd.read_csv("df.csv", index_col=0)
    df["report_date"] = pd.to_datetime(df["report_date"])

    contracts = []
    for cont_id in df.contract_id.unique():
        sl = df[df["contract_id"] == cont_id]
        sl.sort_values("report_date", inplace=True)
        sl["report_num"] = np.arange(len(sl))
        sl["contract_length"] = len(sl)
        contracts.append(sl)
    df = pd.concat(contracts, axis=0)
    df["analytic_lenght"] = df["description"].map(lambda x: x.count("\n"))
    st.session_state.df = df.copy()
else:
    df = st.session_state.df.copy()


class Searcher:
    def __init__(self, search_list):
        self.search_list = search_list

    def __call__(self, searchitem):
        res = []
        for s in self.search_list:
            if str(searchitem) in str(s):
                res.append(s)
        return res


def predict_to_color(p):
    if p < 0.3:
        return "green"
    elif p > 0.65:
        return "red"
    return "orange"


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


df["style_color"] = df["predicts"].map(predict_to_color)

with st.sidebar:
    st.markdown("### TRADOLOGY")
    st.write("Сервис по аналитике подрядчиков в строительном бизнесе.")
    pages_tree = sac.menu(
        items=[
            sac.MenuItem("Сводная"), sac.MenuItem("Аналитика"), sac.MenuItem("Информация об исполнителях"),
        ],
        open_all=True,
        color="red",
    )


def color_rows(row):
    color_dct = {"red": "#FF9C9C", "green": "#A4FF9C", "orange": "#FFE99C"}
    if not pd.isna(row['style_color']):
        return ['']*3 + [f'background-color: {color_dct[row["style_color"]]}'] + ['']*(len(row)-4)
    return [''] * len(row)


if pages_tree == "Сводная":
    st.markdown("#### Сводная информация по выборке")
    st.write("В этом разделе собран информационный дашборд с основной информацией о тестовой выборке. Более подробную инфромацию по каждому отчёту можно узнать в разделе *Аналитика*.")
    col1, col2 = st.columns((4, 6))

    sorting = col1.selectbox("Отсортировать", options=["Нет", "Объёму аналитики", "Возрастанию вероятности срыва",
                                                       "Убыванию вероятности срыва", "Длине контракта",
                                                       "Дате отчёта"])

    min_date, max_date = df["report_date"].max(), df["report_date"].min()
    start_date, end_date = col2.select_slider("Выберите временной промежуток для аналитики",
                                              options=df["report_date"].sort_values(),
                                              value=(min_date, max_date))
    df = df[(df["report_date"] > start_date) & (df["report_date"] < end_date)]

    if sorting == "Дате отчёта":
        df.sort_values("report_date", inplace=True, ascending=False)
    elif sorting == "Длине контракта":
        df.sort_values("contract_length", inplace=True, ascending=False)
    elif sorting == "убыванию вероятности срыва":
        df.sort_values("predicts", inplace=True, ascending=False)
    elif sorting == "Возрастанию вероятности срыва":
        df.sort_values("predicts", inplace=True, ascending=True)
    elif sorting == "Объёму аналитики":
        df.sort_values("analytic_lenght", inplace=True, ascending=False)

    df.sort_values("analytic_lenght", inplace=True, ascending=False)
    styled_df = df.iloc[:2000].style.apply(color_rows, axis=1)
    st.dataframe(styled_df)

    col1, col2 = st.columns((4, 7))
    fig = px.histogram(df["predicts"], title="Распределение вероятностей срыва в выборке",
                       color_discrete_sequence=["#AECFF5"])
    col1.plotly_chart(fig)

    df.sort_values("report_date", inplace=True)
    df["predicts_mean"] = None
    df["predicts_max"] = None
    df["predicts_min"] = None

    for d in df["report_date"].unique():
        df.loc[df["report_date"] == d, "predicts_mean"] = df[df["report_date"] == d]["predicts"].mean()
        df.loc[df["report_date"] == d, "predicts_std"] = df[df["report_date"] == d]["predicts"].min()

    x, y1, y2 = df["report_date"], df["predicts_mean"].tolist(), (df["predicts_mean"] + df["predicts_std"]).tolist()
    y1 = running_mean(y1, 3).tolist() + y1[-2:]
    y2 = running_mean(y2, 3).tolist() + y2[-2:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Средняя вероятность срыва'))
    fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='Отклонение'))
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        title="\t\t\t\t Средняя вероятность срыва по датам"
    )

    col2.plotly_chart(fig)

if pages_tree == "Аналитика":
    st.markdown("#### Аналитика по контрактам и отчётам")
    searcher = Searcher(df.contract_id.unique())
    selected_value = st_searchbox(
        searcher,
        placeholder="Выберите id контракта",
        default=""
    )
    if selected_value == "":
        cont_id = 3913
    else:
        cont_id = selected_value
    st.markdown(f"Контракт: **{cont_id}**")

    df = df[df["contract_id"] == cont_id]
    df.sort_values("report_date", inplace=True, ascending=False)
    min_date, max_date = df["report_date"].max(), df["report_date"].min()
    start_date, end_date = st.select_slider("Выберите временной промежуток для аналитики",
                                              options=df["report_date"].sort_values(),
                                              value=(min_date, max_date))
    df = df[(df["report_date"] > start_date) & (df["report_date"] < end_date)]

    col1, col2 = st.columns((4, 6))
    rep = col1.selectbox("Выберите отчёт", df["report_date"],)

    if len(df) == 0:
        st.error("Ошибка в выборе временного промежутка. Не найдено ни одного подходящего отчёта для этого контракта")
    else:
        if len(df) > 1:
            col1, col2 = st.columns((5, 7))

            x, y = df["report_date"], df["predicts"].tolist()
            y = running_mean(y, 3).tolist() + y[-2:]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                                     name='Вероятность срыва'))
            fig.add_trace(go.Scatter(x=x, y=y + np.std(y),
                                     mode='lines', name='Отклонение'))
            fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                ),
                title="\t\t\t\t Вероятности срыва"
            )
            col2.plotly_chart(fig)

            fig = px.histogram(df["predicts"], title="Распределение вероятностей срыва в контракте",
                               color_discrete_sequence=["#AECFF5"])
            col1.plotly_chart(fig)

        dscr = df[df["report_date"] == rep].iloc[0]["description"]
        st.markdown(dscr.replace("\n", "<br><br>"), unsafe_allow_html=True)

if pages_tree == "Информация об исполнителях":
    st.markdown("#### Информация об исполнителях")
    st.write("Тут можно сравнить исполнителей по индексам доверия к ним: метрика оценки качества исполнителя, основанная на выполненных исполнителем контрактах и контрактах исполнителей, с которыми он связан по матрице расстояний исполнителей.")

    contractor_idxes = pd.read_csv("contractor_indexes_fixed.csv")
    contractor_idxes["success_index"] = contractor_idxes["success_index"].map(lambda x: x if x < 1 else 1)
    searcher = Searcher(contractor_idxes.contractor_id.unique())
    selected_value = st_searchbox(
        searcher,
        placeholder="Выберите id исполнителя",
        default=""
    )
    if selected_value == "":
        selected_value = 683
    st.markdown(f"Исполнитель с id **{selected_value}**")
    succ_idx = round(contractor_idxes[contractor_idxes["contractor_id"] == selected_value].iloc[0]["success_index"], 2)
    if succ_idx > 0.6:
        color = "green"
    else:
        color = "red"
    st.markdown(f"Индекс уверенности в поставщике: :{color}[{succ_idx}]")
    if succ_idx > 0.6:
        st.write("Это хороший индекс уверенности, данный исполнитель имеет хорошую репутацию выполнения работ.")
    else:
        st.write("Это низкий индекс уверенности, данный исполнитель не отличается стабильностью в выполнении заказов.")
    sl = df[df["идентификатор подрядчика"] == selected_value]
    if len(sl) > 0:
        nice_contracts = sl[sl["predicts"] > 0.65]["contract_id"].map(lambda x: str(x)).unique()
        st.write("Успешные контракты исполнителя:", ", ".join(nice_contracts.tolist()))

        bad_contracts = sl[sl["predicts"] < 0.65]["contract_id"].map(lambda x: str(x)).unique()
        st.write("Неуспешные контракты исполнителя:", ", ".join(bad_contracts.tolist()))

    st.markdown("-----")
    worst_contractors = contractor_idxes[contractor_idxes["success_index"] == contractor_idxes["success_index"].min()]
    best_contractors = contractor_idxes[contractor_idxes["success_index"] == contractor_idxes["success_index"].max()]
    col1, col2 = st.columns((5, 5))

    col1.write("Исполнители с самым низким индексом уверенности")
    col1.write(worst_contractors)

    col2.write("Исполнители с самым вызоким индексом уверенности")
    col2.write(best_contractors)