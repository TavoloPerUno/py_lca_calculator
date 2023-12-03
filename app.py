import streamlit as st
import pandas as pd
import numpy as np
import hvplot.pandas
import holoviews as hv

import yaml

dct_config = yaml.safe_load(open("config.yaml", "r"))
study_name = next(iter(dct_config))
dct_study_config = next(iter(dct_config.values()))

st.set_page_config(page_title=study_name, )
st.title(study_name)
st.divider()
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def load_data():
    pdf_coeff = pd.DataFrame(dct_study_config["covariate_coefficients"])
    pdf_intercept = (
        pd.DataFrame(dct_study_config["class_intercepts"], index=[0])
        .T.reset_index()
        .rename(columns={0: "Intercepts", "index": "Latent Classes"})
    )
    pdf_probabilities = pd.DataFrame(dct_study_config["outcome_probabilities"])
    pdf_config = pd.concat(
        {
            "": pdf_intercept["Latent Classes"],
            "Beta Estimates": pd.concat(
                [pdf_intercept[["Intercepts"]], pdf_coeff], axis=1
            ),
            "Outcome Probabilities": pdf_probabilities,
        },
        axis=1,
    )
    return pdf_config


def get_class_and_outcome_probabilities(selected_covariates):
    pdf_coefficients = pdf_study_config["Beta Estimates"]
    pdf_coefficients = pdf_coefficients.assign(
        log_odds=pdf_coefficients[["Intercepts"] + selected_covariates].sum(axis=1)
    )
    pdf_coefficients.loc[0, "log_odds"] = 0
    pdf_coefficients = pdf_coefficients.assign(
        class_probabilities=np.exp(pdf_coefficients["log_odds"])
        / np.exp(pdf_coefficients["log_odds"]).sum(axis=0)
    )
    return pdf_coefficients


pdf_study_config = load_data()
dct_selected_covariates = {}
with st.container():
    st.write(f"Select applicable {dct_study_config['covariate_label']}")
    covariates = list(
        pdf_study_config["Beta Estimates"].columns.difference(["Intercepts"])
    )
    dct_cols = {}
    dct_cols[0], dct_cols[1], dct_cols[2] = st.columns(3)
    for idx, covariate in enumerate(covariates):
        with dct_cols[idx % 3]:
            dct_selected_covariates[covariate] = st.radio(
                covariate, ["Yes", "No", "Unknown"], index=None
            )

with st.container():
    if all(
        bool(dct_selected_covariates[covariate])
        for covariate in dct_selected_covariates
    ):
        st.divider()
        selected_covariates = [
            covariate
            for covariate in dct_selected_covariates
            if dct_selected_covariates[covariate] == "Yes"
        ]
        pdf_class_probabilities = get_class_and_outcome_probabilities(
            selected_covariates
        )
        pdf_class_probabilities = pdf_study_config[""].merge(
            pdf_class_probabilities[["class_probabilities"]],
            left_index=True,
            right_index=True,
            how="inner",
        )
        pdf_predicted_class = pdf_study_config.iloc[
            pdf_class_probabilities["class_probabilities"].idxmax(),
        ]
        col1, col2 = st.columns(2)
        with col1:
            outcome_probabilities = "\n".join(
                [
                    f"- {idx}: {value * 100} %"
                    for idx, value in pdf_predicted_class[
                        "Outcome Probabilities"
                    ].items()
                ]
            )
            st.markdown(
                f"""
            Predicted latent class 
            is {pdf_predicted_class['']['Latent Classes']} with 
            outcome probabilities \n{outcome_probabilities}
            """
            )
        with col2:
            st.write(
                hv.render(
                    pdf_class_probabilities.set_index("Latent " "Classes")
                    .T.hvplot.bar(
                        responsive=True,
                        stacked=True,
                        height=300,
                        xlabel="",
                        xaxis=None,
                        ylabel="Class probabilities",
                        title="Predicted class probabilities",
                        sizing_mode='scale_both'
                    )
                    .opts(default_tools=["save"]),
                    backend="bokeh",
                )
            )

st.divider()
pdf_outcome_probabilities = pdf_study_config.set_index(("", "Latent Classes"))[
    "Outcome Probabilities"
]
pdf_outcome_probabilities.index.name = "Latent Classes"

st.write(
    hv.render(
        pdf_outcome_probabilities.hvplot.barh(
            responsive=True,
            height=500,
            rot=90,
            xlabel="Latent Classes, Outcome",
            title="Probability of outcome, given latent class",
            sizing_mode='scale_both'
        ).opts(default_tools=["save"]),
        backend="bokeh",
    )
)
