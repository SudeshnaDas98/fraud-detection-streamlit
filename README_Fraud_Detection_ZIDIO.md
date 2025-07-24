
# Real-Time Financial Fraud Detection System

A complete end-to-end fraud detection system developed during our internship at **ZIDIO**. This solution uses a combination of **supervised** and **unsupervised** machine learning models, integrated with **real-time monitoring**, **alerting**, and **visual dashboards** to detect and flag fraudulent financial transactions.

---

## Project Highlights

- Achieved **99%+ accuracy** using advanced ML algorithms: `XGBoost`, `LightGBM`, and `Random Forest`.
- Integrated **real-time fraud prediction** pipeline with **Apache Kafka**.
- Developed an interactive **Streamlit dashboard** for live monitoring.
- Enabled **email alert notifications** for suspicious/high-risk transactions.
- Added anomaly detection using `Isolation Forest` and `PCA`.
- Incorporated **graph-based fraud ring visualization** using `NetworkX`.

---

## Tech Stack

| Component           | Technology               |
|---------------------|--------------------------|
| **Language**         | Python                   |
| **ML Frameworks**    | Pandas, Scikit-learn, XGBoost, LightGBM |
| **Real-Time Engine** | Apache Kafka             |
| **Dashboard**        | Streamlit                |
| **Alerting System**  | SMTP (Email Integration) |
| **Graph Analysis**   | NetworkX                 |

---

## Models Implemented

| Model                         | Type        | Highlights                                      |
|-------------------------------|-------------|-------------------------------------------------|
| **Random Forest**             | Supervised  | High accuracy, robustness to overfitting        |
| **XGBoost**                   | Supervised  | Best overall model performance                  |
| **LightGBM**                  | Supervised  | Fast training and scoring with high precision   |
| **Isolation Forest**          | Unsupervised| Detects anomalies not seen during training      |
| **PCA (Reconstruction Error)**| Unsupervised| Captures irregular transactions via error size  |
| **Graph-Based Detection**     | Conceptual  | Visualizes fraudulent clusters and connections  |

---

## Real-Time Monitoring Dashboard

- Built using **Streamlit** for a user-friendly experience.
- Displays real-time logs of processed transactions.
- Shows **fraud vs. legit** counts using bar charts.
- Auto-refreshes every 5 seconds.
- Visual indicators highlight suspicious patterns and decisions.

---

## Email Alert System

- Configured using **SMTP** to send **real-time email alerts**.
- Automatically triggers on detection of fraudulent transactions.
- Alerts contain transaction amount, risk score, and timestamp.
- Ensures immediate awareness and response to high-risk events.

---

## Future Enhancements

- Integration of **AI-driven risk scoring system**.
- **Blockchain-based** transaction tracking and validation.
- **Mobile app** for fraud alerts and monitoring on the go.

---

## Conclusion

This real-time financial fraud detection system represents a **comprehensive and forward-thinking approach** to combating digital financial crimes.

By integrating both **supervised and unsupervised** ML models, the system effectively detects known fraud patterns and adapts to **emerging, previously unseen threats**.

- Apache Kafka enables seamless **real-time data streaming**.
- Streamlit provides **intuitive visual dashboards** for analysts.
- Features like **automated email alerts** and **graph-based fraud ring detection** enhance both usability and insight.

Developed during our internship at **ZIDIO**, this project demonstrates strong technical capability and **real-world applicability** of ML in financial services. Its modular design ensures **easy integration** with future technologies such as **AI-powered scoring** or **blockchain validation**.

> In essence, this project delivers a practical and scalable solution to the growing challenge of financial fraud.

