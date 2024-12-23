# <p align='center'>Risks estimation in development [(AI Challenge 2024)](https://aiijc.com)</h1>

### Intro to domain: 
Its very important to carefully select reliable contractor if you are development group like [`Samolet`](https://samolet.ru/). Wrong choice may cost you a lot of time and money if contractor will break the deal. 

### Data:
- Contracts starting from the begging of 2023. Each contract consists of several reports(each report is being created each week). Reports described by around 2 hundred features. Data have tabular view (**sample**: `report_id`, `contract_id`, `contractor_id` ...)
- Also we have **graph of contractors relations** (u,v,w), where w denotes strength of the connection between `contractor_u` and `contractor_v`.

### Task formalization:
*We need predict the probability of the **contract`s default in period of 6 months starting from this report**. In the last competition stage we also had to interpret predicted probabilty*

### Key features of solution:
- Special model for classifying whole contract. According to given dataset less than 5% of the contracts have both the 1 and 0 target (in the beggining of the contract probabilty should be small but than should increase) so it makes sense to build model for classifying if contract will fail or not.
- Variety of generated features. We generated different features starting from tsfresh and custom binarization to graph contractor embeddings and empirical features like PageRank.
- Robustness of solution. We did custom features selection and through out features that were creating bias in train 
  
### Repository structure
- <ins>**notebooks/training**</ins> - all notebooks with various training methods for solving target task (RNNs, boostings...)
- <ins>**notebooks/features**</ins> - all notebooks with feature generation code (GNN embeddings, tsfresh...)
- <ins>**notebooks/research**</ins> - all notebooks with experiments
- <ins>**app**</ins> - full working streamlit app with best solution (p.s. not last stage version yet)
- <ins>**data**</ins> - all data including generated features and other stuff  
- <ins>**development.pdf**</ins> - presentation of solution for AI Challenge contest

### Additional info
- *Our solution took 2nd place in AI Challenge*

### Contacts
- **[seyolax](https://t.me/seyolax)**
- **[danisdinm@gmail.com](mailto:danisdinm@gmail.com)**
