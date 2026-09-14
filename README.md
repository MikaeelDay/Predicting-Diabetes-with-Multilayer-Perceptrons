# Predicting Diabetes with Multilayer Perceptron (MLP)

پیش‌بینی احتمال ابتلا به دیابت با استفاده از یک شبکه‌ی عصبی Multilayer Perceptron (MLP)، بر اساس داده‌های پزشکی بیماران.

این پروژه با الهام از فصل دوم کتاب *Neural Network Projects with Python* نوشته‌ی James Loy ساخته شده است.

---

## 📋 درباره پروژه

هدف این پروژه، پیش‌بینی این موضوع است که آیا یک بیمار طی ۵ سال آینده به دیابت مبتلا می‌شود یا نه؛ بر اساس چند اندازه‌گیری پزشکی ساده مانند سطح گلوکز خون، فشار خون، BMI و... .

برای این کار از دیتاست معروف **Pima Indians Diabetes Dataset** استفاده شده که توسط National Institute of Diabetes and Digestive and Kidney Diseases گردآوری و در Kaggle منتشر شده است.

## 🗂 دیتاست

دیتاست شامل ۹ ستون است:

| ستون | توضیح |
|---|---|
| `Pregnancies` | تعداد بارداری‌های قبلی |
| `Glucose` | غلظت گلوکز پلاسما |
| `BloodPressure` | فشار خون دیاستولیک |
| `SkinThickness` | ضخامت چین پوستی سه‌سر بازو |
| `Insulin` | غلظت انسولین سرم خون |
| `BMI` | شاخص توده بدنی |
| `DiabetesPedigreeFunction` | امتیاز مربوط به استعداد ژنتیکی ابتلا به دیابت |
| `Age` | سن (سال) |
| `Outcome` | برچسب هدف: ۱ برای ابتلا به دیابت، ۰ در غیر این صورت |

مجموعاً ۷۶۸ رکورد وجود دارد که حدود ۶۵٪ آن‌ها کلاس ۰ (بدون دیابت) و ۳۵٪ کلاس ۱ (دیابت) هستند.

## ⚙️ پیش‌پردازش داده

- مقادیر **صفر** در ستون‌های `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` به‌عنوان مقدار گم‌شده (Missing Value) در نظر گرفته و با **میانگین** مقادیر غیر گم‌شده جایگزین شدند (توجه: مقدار صفر در ستون `Pregnancies` معتبر است و دست‌نخورده باقی می‌ماند).
- تمام ویژگی‌های عددی با استفاده از `sklearn.preprocessing.scale` استانداردسازی شدند (میانگین صفر و واریانس واحد) تا مقیاس متفاوت متغیرها (مثلاً `Insulin` تا ۸۴۶ در برابر `DiabetesPedigreeFunction` تا ۲.۴۲) مانع یادگیری درست شبکه نشود.
- داده به سه بخش تقسیم شد:
  - **Training set**: برای آموزش مدل
  - **Validation set**: برای تنظیم هایپرپارامترها
  - **Testing set**: برای ارزیابی نهایی و بی‌طرفانه

تقسیم‌بندی به‌صورت تصادفی و طی دو مرحله انجام شد: ابتدا ۸۰٪/۲۰٪ برای train/test و سپس ۸۰٪/۲۰٪ از داده‌ی train برای train/validation.

## 🧠 معماری مدل

یک MLP ساده با دو لایه‌ی مخفی:

```
Input Layer (8 نورون)  →  Hidden Layer 1 (32 نورون, ReLU)
                        →  Hidden Layer 2 (16 نورون, ReLU)
                        →  Output Layer (1 نورون, Sigmoid)
```

- **ReLU** برای لایه‌های مخفی (استاندارد رایج در شبکه‌های عمیق)
- **Sigmoid** برای لایه‌ی خروجی، چون مسئله از نوع طبقه‌بندی دودویی (Binary Classification) است

### تنظیمات آموزش

| پارامتر | مقدار |
|---|---|
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Metric | Accuracy |
| Epochs | 200 |

## 📊 نتایج

| Metric | مقدار |
|---|---|
| Training Accuracy | ۹۱.۸۵٪ |
| Testing Accuracy | ۷۸.۵۷٪ |

### Confusion Matrix

|  | پیش‌بینی: بدون دیابت | پیش‌بینی: دیابت |
|---|---|---|
| **واقعی: بدون دیابت** | True Negative: 81 | False Positive: 14 |
| **واقعی: دیابت** | False Negative: 19 | True Positive: 40 |

همچنین منحنی **ROC** برای ارزیابی توانایی مدل در تفکیک دو کلاس رسم شده که نشان‌دهنده‌ی عملکرد خوب مدل در مقایسه با حالت تصادفی است.

## 🛠 پیش‌نیازها

```
Python 3.x
pandas 0.23.4
numpy 1.15.2
matplotlib 3.0.2
seaborn 0.9.0
scikit-learn 0.20.2
Keras 2.2.4
```

## 🚀 نحوه اجرا

```bash
# کلون کردن ریپازیتوری
git clone <آدرس این ریپازیتوری>
cd <نام-پوشه>

# نصب پیش‌نیازها
pip install -r requirements.txt

# اجرای مدل
python main.py

# رسم نمودارهای تحلیل اکتشافی داده (EDA)
python visualize.py
```

## 📁 ساختار پروژه

```
.
├── diabetes.csv       # دیتاست
├── main.py             # کد اصلی ساخت، آموزش و ارزیابی مدل
├── utils.py            # توابع کمکی (پیش‌پردازش و ...)
├── visualize.py         # کد تحلیل اکتشافی و visualization داده
└── README.md
```

## 📚 منبع

این پروژه بر اساس فصل دوم کتاب زیر پیاده‌سازی شده است:

> James Loy, *Neural Network Projects with Python*, Packt Publishing.

دیتاست اصلی: [Pima Indians Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

## 📝 لایسنس

این پروژه صرفاً با هدف آموزشی ساخته شده است.
