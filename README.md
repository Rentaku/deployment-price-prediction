## Deployment-motorcycle-price-prediction

gcloud builds submit --tag gcr.io/rentaku-capstone/price_predictmoto

gcloud run deploy --image gcr.io/rentaku-capstone/price_predictmoto --platform managed


## Deployment-motorcycle-price-prediction

gcloud builds submit --tag gcr.io/rentaku-capstone/price_predictcar

gcloud run deploy --image gcr.io/rentaku-capstone/price_predictcar --platform managed
