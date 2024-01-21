"""Streamlit app to compute LCA class probabilities based on covariate values
"""
from pathlib import Path
import streamlit as st
import pandas as pd
import hvplot.pandas
import holoviews as hv
import yaml

with open("config.yaml", "r", encoding="utf-8") as config_file:
    dct_config = yaml.safe_load(config_file)
study_name = next(iter(dct_config))
dct_study_config = next(iter(dct_config.values()))
st.set_page_config(page_title=study_name, page_icon=":stethoscope:")
st.title(study_name)
st.divider()
with open(Path("static/style.css"), encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_data
def get_model_estimates():
    """Loads model estimates from config file"""
    pdf_coeff = pd.DataFrame(dct_study_config["covariate_class_probabilities"])
    pdf_prior_class_probabilities = (
        pd.DataFrame(dct_study_config["prior_probabilities"], index=[0])
        .T.reset_index()
        .rename(columns={0: "Prior Probabilities", "index": "Latent Classes"})
    )

    pdf_outcomes = pd.DataFrame(dct_study_config["outcome_rates"]).merge(
        pdf_prior_class_probabilities[["Latent Classes"]],
        left_index=True,
        right_index=True,
        how="inner",
    )

    pdf_weights_and_probabilities = pd.concat(
        {
            "Model Estimates": pd.concat(
                [
                    pdf_prior_class_probabilities[
                        ["Latent Classes", "Prior Probabilities"]
                    ],
                    pdf_coeff,
                ],
                axis=1,
            ).set_index("Latent Classes"),
            "Outcome Rates": pdf_outcomes.set_index("Latent Classes"),
        },
        axis=1,
    )
    return pdf_weights_and_probabilities


@st.cache_data
def compute_class_probabilities(existing, observed):
    """
    This function will calculate class probabilities for K classes using the
    below formula:

    .. math::
        \Pr (j | x_{obs}) = \frac{\hat{\eta_j}\prod_{i=1}^{P}[\hat{\pi}_{ij}^{x_i}(1-\hat{\pi}_{ij})^{1 - x_i}]^{r_i} }{\hat{f}(x_{obs})}\newline
        \text{where } \hat{f}(x_{obs}) = \sum_{j=1}^{K}\hat{\eta}_j \prod_{i=1}^{P}[\hat{\pi}_{ij}^{x_i}(1-\hat{\pi}_{ij})^{1 - x_i}]^{r_i} \newline
        K = \text{Number of classes} \newline
        x_i = \text{binary indicator variables used in the LCA model } (i = 1, 2, . . . , p) \newline
        \hat{\pi}_{ij} = \text{Estimate of the probability that }x_i = 1 \text{ for an individual in Class j} \newline
        x_{obs} = \text{vector of observed variables} \newline
        r_i = \text{indicator variable taking the value 1 when }x_i \text{ is observed and 0 otherwise}

    """
    pdf_coefficients = pdf_study_config["Model Estimates"]
    print(pdf_coefficients)

    pdf_coefficients = pdf_coefficients.assign(
        lca_model=pd.concat(
            [
                pdf_coefficients[
                    ["Prior Probabilities"]
                    + [col for col in observed if col in existing]
                ],
                1 - pdf_coefficients[[col for col in observed if col not in existing]],
            ],
            axis=1,
        ).prod(axis=1)
    )
    print(pdf_coefficients)
    pdf_coefficients = (
        (
            pdf_coefficients.reset_index().groupby("Latent Classes")["lca_model"].sum()
            / pdf_coefficients["lca_model"].sum()
        )
        .to_frame()
        .rename(columns={"lca_model": "class_probabilities"})
    )
    print(pdf_coefficients)
    return pdf_coefficients


pdf_study_config = get_model_estimates()
dct_selected_covariates = {}
with st.container():
    st.write(f"Select applicable {dct_study_config['covariate_label'].lower()}")
    covariates = list(
        pdf_study_config["Model Estimates"].columns.difference(["Prior Probabilities"])
    )
    dct_cols = {}
    dct_cols[0], dct_cols[1], dct_cols[2] = st.columns(3)
    for idx, covariate in enumerate(covariates):
        with dct_cols[idx % 3]:
            dct_selected_covariates[covariate] = st.radio(
                covariate, ["Yes", "No", "Unknown"], index=2
            )

with st.container():
    if all(
        bool(covariate_value)
        for covariate, covariate_value in dct_selected_covariates.items()
    ):
        st.divider()
        existing_covariates = [
            covariate
            for covariate, covariate_value in dct_selected_covariates.items()
            if covariate_value == "Yes"
        ]
        observed_covariates = [
            covariate
            for covariate, covariate_value in dct_selected_covariates.items()
            if covariate_value != "Unknown"
        ]
        pdf_class_probabilities = compute_class_probabilities(
            existing_covariates, observed_covariates
        )
        pdf_predicted_class = pdf_study_config.loc[
            pdf_class_probabilities["class_probabilities"].idxmax(),
        ]
        col1, col2 = st.columns(2)
        with col1:
            outcome_rates = "\n".join(
                [
                    f"- {idx}: {round(value, 2)}"
                    for idx, value in pdf_predicted_class["Outcome Rates"].items()
                ]
            )
            st.markdown(
                f"""Predicted latent class is **{pdf_predicted_class.name}** 
                with outcome rates ({dct_study_config['outcome_rate_unit']}) 
                \n{outcome_rates}"""
            )
        with col2:
            st.bokeh_chart(
                hv.render(
                    pdf_class_probabilities.T.hvplot.bar(
                        responsive=True,
                        stacked=True,
                        height=300,
                        xlabel="",
                        xaxis=None,
                        ylabel="Class probabilities",
                        title="Predicted class probabilities",
                        sizing_mode="scale_both",
                    ).opts(default_tools=["save"]),
                    backend="bokeh",
                )
            )

st.divider()
pdf_covariate_probabilities = pdf_study_config["Model Estimates"].drop(
    columns=["Prior Probabilities"]
)

st.bokeh_chart(
    hv.render(
        pdf_covariate_probabilities.hvplot.barh(
            responsive=True,
            height=800,
            ylabel="Probability",
            xlabel=f"Latent Classes, {dct_study_config['covariate_label']}",
            title=f"Probability of {dct_study_config['covariate_label']}, "
            "given latent class",
            sizing_mode="scale_both",
        ).opts(default_tools=["save"]),
        backend="bokeh",
    )
)

st.divider()
st.bokeh_chart(
    hv.render(
        pdf_study_config["Outcome Rates"]
        .hvplot.barh(
            responsive=True,
            height=800,
            ylabel=f"Outcome rates ({dct_study_config['outcome_rate_unit']})",
            xlabel=f"Latent Classes, Outcome rates",
            title=f"Outcome rates ({dct_study_config['outcome_rate_unit']}) "
            f"for each latent class",
            sizing_mode="scale_both",
        )
        .opts(default_tools=["save"]),
        backend="bokeh",
    )
)
