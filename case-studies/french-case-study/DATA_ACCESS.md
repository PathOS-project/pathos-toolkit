# Data Access Information

## Data Availability

**Zenodo DOI**: [10.5281/zenodo.17258940](https://doi.org/10.5281/zenodo.17258940)

### What's Available in Zenodo
This dataset contains the value of the Open Science Access Advantage indicator calculated for all the ressources accessed on the French Open Science platform HAL.archives-ouvertes.fr, between September 2023 et August 2024.

The data have been obtained through a partnership with the platform which provided us all the logs of its server for the period above. Though a parnership with IPinfo.io, we categorized these access geographically and by economic sector.

The Open Science Access Advantage compares the ratio of accesses directed to OS resources over the total number of views, to the ratio of open publications among the relevant resources. This indicator has been computed for the entire HAL platform, for a single discipline (considering all publications associated with it), a single country or sector (considering all the publications accessed from it), as well as for any combination of thereof.

The indicator is operationalized through a relatively straightforward metric comparing the ratio to which OS resources are accessed to the ratio of their availability. More precisely, the metric is calculated as follows

Where:

* SO (stock open) is the number OS resources accessed in a given disciplinary, sectoral and geographical combination.
* SC (stock closed) is the number of non-OS resources accessed in the same combination.
* AO (access open) is the number of views collected by OS resources
* AC (access closed) is the number of views collected by non-OS resources

Basically, the metrics compare the ratio of SO resources (the part of SO over the total of relevant resources) to the ratio of the accesses directed to SO resources (over the total of accesses). Since both the ratios vary between zero and one, their difference varies between one (when most of the available resources are closed and yet most of the accesses are directed to few open ones) and minus one and is minimal (when most of the available resources are open and yet most of the accesses are directed to the few close ones).

As both ratios can be expressed as percentages, the Open Access Advantage indicator can also be expressed as a percentage which is the larger the higher is the the probability of open resources to be consulted compared to closed ones.

## External Data Sources Required

- IPinfo’s Academic Research Program (https://ipinfo.io/use-cases/ip-data-for-academic-research). Book an appointment with a business representative, have the academic research agreement signed, and get access to the API 
- Hal & OpenEdition connection logs. Work with the sysadmin teams to obtain access. 
- OpenAlex API : https://docs.openalex.org/api-entities/works
- LLama3.3  (https://ollama.com/library/llama3.3, https://gitlab.huma-num.fr/path-os/automated-nace-classifier, https://doi.org/10.5281/zenodo.17471199)
- Eurostat NACE Rev. 2.1 (https://ec.europa.eu/eurostat/web/nace)

## Usage

### For Analysis with Provided Data
1. Download processed datasets from Zenodo
2. Run analysis scripts with the provided data

### For Full Reproduction
1. Obtain access to required external data sources
2. See readme.md and D3.4 for full instructions and data pipe line to enrich the raw logs with information about users and ressources which allow for aggregated statistics

## Contact

simon.apartis@cnrs.fr
tommaso.venturini@cnrs.fr
melanie.dulong@cnrs.fr
paul@ouestware.com