import pandas as pd
import numpy as np
from pathlib import Path
from app.db.database import SessionLocal
import app.models as models
from bs4 import BeautifulSoup
import huspacy
import spacy
import unicodedata
import pickle
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

class PredictorService :

    def __init__(self):
        self.nlp = None
        self.df = None

    def lemmatize_text(self, text: str) -> str:
        if(self.nlp is None):
            huspacy.download()
            self.nlp = spacy.load("hu_core_news_lg")

        doc = self.nlp(text.lower())
        return ' '.join(
            token.lemma_ for token in doc
            if not token.is_stop and not token.is_punct and token.is_alpha
        )

    def remove_accents(self, text: str) -> str:
        if isinstance(text, str):
            nfkd_form = unicodedata.normalize('NFKD', text)
            return ''.join([c for c in nfkd_form if not unicodedata.combining(c)]).lower()
        return text

    def lemmatize_data_to_db(self):
        with (SessionLocal() as db):
            articles = db.query(models.Articles).filter(models.Articles.lemmatized_text == None).all()

            for article in articles:
                summary_text = BeautifulSoup(str(article.summary), 'html.parser').get_text()
                full_text = f"{article.title} {summary_text}"

                article.lemmatized_text = self.lemmatize_text(full_text)

            db.commit()

    def preparation_data_to_train(self):
        with (SessionLocal() as db):
            articles = db.query(
                models.Articles.title,
                models.Articles.summary,
                models.Articles.lemmatized_text,
                models.Articles.category,
            ).all()
            df = pd.DataFrame(articles, columns=["title", "summary", "lemmatized_text", "category"])
            df['category'] = df['category'].apply(self.remove_accents)
            df['summary'] = df['summary'].apply(lambda x: BeautifulSoup(str(x), 'html.parser').get_text())
            df.dropna(inplace=True)
            df['category'] = df['category'].replace({
                'global': 'kulfold',
                'hirtvkulfold': 'kulfold',
                'kulpol': 'kulfold',
                'nagyvilag': 'kulfold',
                'vilag': 'kulfold',
                'celeb': 'bulvar',
                'sztarok': 'bulvar',
                'sztarvilag': 'bulvar',
                'kultur': 'kultura',
                'teve': 'bulvar',
                'techtud': 'tudomany',
                'techbazis': 'tudomany',
                'tech-tudomany': 'tudomany',
                'tech': 'tudomany',
                'itthon': 'belfold',
                'belpol': 'belfold',
                'politika': 'belfold',
                'nemzetkozi-gazdasag': 'gazdasag',
                'uzlet': 'gazdasag',
                'penz': 'gazdasag',
                'bank': 'gazdasag',
                'befektetes': 'gazdasag',
                'deviza': 'gazdasag',
                'vilaggazdasag-magyar-gazdasag': 'gazdasag',
                'elet-stilus': 'eletmod',
                'ferfiaknak': 'eletmod',
                'test-es-lelek': 'eletmod',
                'bunugyek': 'baleset-bunugy',
                'futball': 'sport',
                'foci': 'sport',
                'szorakozas': 'bulvar',
            })
            df = df[~df['category'].isin(['2023', 'hirek', 'hirado', 'aktualis', 'mindekozben'])]
            category_counts = df['category'].value_counts()
            valid_categories = category_counts[category_counts >= 300].index
            df = df[df['category'].isin(valid_categories)]
            self.df = df

    def train_xgboost(self):
        self.preparation_data_to_train()
        df = self.df
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

        df['combined'] =  df['title'] + ' ' + df['summary'] + ' ' + df['lemmatized_text']
        X = vectorizer.fit_transform(df['combined'])

        le = LabelEncoder()
        y = le.fit_transform(df['category'])

        classes = np.unique(y)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
        weight_dict = dict(zip(classes, class_weights))
        sample_weights = np.array([weight_dict[label] for label in y])

        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, sample_weights, test_size=0.2, stratify=y, random_state=42
        )

        params = {
            'objective': 'multi:softmax',
            'num_class': len(np.unique(y_train)),
            'eval_metric': 'mlogloss',
            'n_jobs': 1,
            'n_estimators': 301,
            'max_depth': 9,
            'learning_rate': 0.24033105011399966,
            'subsample': 0.8219754756042266,
            'colsample_bytree': 0.9863041453317289,
            'gamma': 3
        }

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, sample_weight=w_train)

        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=le.classes_))

    def train_neural_network(self):
        self.preparation_data_to_train()
        df = self.df
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

        df['combined'] =  df['title'] + ' ' + df['summary'] + ' ' + df['lemmatized_text']
        X = vectorizer.fit_transform(df['combined'])

        le = LabelEncoder()
        y = le.fit_transform(df['category'])
        y_cat = to_categorical(y)

        X_train, X_temp, y_train, y_temp = train_test_split(X, y_cat, stratify=y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
        weight_dict = dict(enumerate(class_weights))

        model = Sequential([
            Dense(64, activation='relu', input_shape=(X.shape[1],)),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dense(y_cat.shape[1], activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

        model.fit(
            X_train.toarray(), y_train,
            validation_data=(X_val.toarray(), y_val),
            epochs=10,
            batch_size=64,
            class_weight=weight_dict,
            callbacks=[early_stop]
        )

        loss, acc = model.evaluate(X_test.toarray(), y_test)
        print(f"Pontosság: {acc:.3f}")

        with (SessionLocal() as db):
            better_exists =  db.query(models.Training).filter(models.Training.accuracy > acc).first()

            if better_exists:
                training = models.Training(
                    model_name='neural_network',
                    training_date=pd.Timestamp.now(),
                    accuracy=acc,
                    active=False
                )
                db.add(training)
                db.commit()
            else:
                db.query(models.Training).filter(models.Training.active == True).update({"active": False})

                training = models.Training(
                    model_name='neural_network',
                    training_date=pd.Timestamp.now(),
                    accuracy=acc,
                    active=True
                )
                db.add(training)
                db.commit()
                ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
                model.save(ARTIFACTS_DIR / "category_predicter.keras")
                with open(ARTIFACTS_DIR / "vectorizer.pkl", "wb") as f:
                    pickle.dump(vectorizer, f)
                with open(ARTIFACTS_DIR / "label_encoder.pkl", "wb") as f:
                    pickle.dump(le, f)

    def predict(self, text: str, leman: str = None):
        model = load_model(ARTIFACTS_DIR / "category_predicter.keras")
        with open(ARTIFACTS_DIR / "vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open(ARTIFACTS_DIR / "label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        if leman is None:
            leman = self.lemmatize_text(text)
        combined = f'{text} {leman}'
        X = vectorizer.transform([combined]).toarray()
        y_pred = model.predict(X)
        pred_label = np.argmax(y_pred, axis=1)
        category = le.inverse_transform(pred_label)[0]
        return category







