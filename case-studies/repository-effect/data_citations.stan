data {
    int<lower=1> N;
    int<lower=1> N_repositories;
    int<lower=1> N_sources;
    int<lower=1> N_years;
    int<lower=1> N_fields;

    array[N] int<lower=0,upper=1> is_cited;
    array[N] int<lower=1,upper=N_repositories> repository;
    array[N] int<lower=1,upper=N_sources> source;
    array[N] int<lower=1,upper=N_years> year;
    array[N] int<lower=1,upper=N_fields> field;
}
parameters {
    real intercept;
    sum_to_zero_vector[N_repositories] repository_effect;
    sum_to_zero_vector[N_sources] source_effect;
    sum_to_zero_vector[N_years] year_effect;
    sum_to_zero_vector[N_fields] field_effect;

}
model {
    // Priors
    intercept ~ normal(0, 5);
    repository_effect ~ normal(0, sqrt(N_repositories / (N_repositories - 1.0)));
    source_effect ~ normal(0, sqrt(N_sources / (N_sources - 1.0)));
    year_effect ~ normal(0, sqrt(N_years / (N_years - 1.0)));
    field_effect ~ normal(0, sqrt(N_fields / (N_fields - 1.0)));

    // Main model
    is_cited ~ bernoulli_logit(intercept +
                               repository_effect[repository] +
                               source_effect[source] +
                               year_effect[year] +
                               field_effect[field]);
}

