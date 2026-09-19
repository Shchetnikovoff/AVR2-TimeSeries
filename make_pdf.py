"""make_pdf.py — сборка PDF-версии отчета AVR-2 для вложения в сдачу."""
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

FONT_DIR = "/System/Library/Fonts/Supplemental/"
pdfmetrics.registerFont(TTFont("AU", FONT_DIR + "Arial Unicode.ttf"))
pdfmetrics.registerFont(TTFont("AUB", FONT_DIR + "Arial Bold.ttf"))

W, H = A4
M = 1.6 * cm

st_title = ParagraphStyle("t", fontName="AUB", fontSize=16, leading=20, spaceAfter=8)
st_h1 = ParagraphStyle("h1", fontName="AUB", fontSize=12, leading=15, spaceBefore=12, spaceAfter=5,
                       textColor=colors.HexColor("#1a1a1a"))
st_p = ParagraphStyle("p", fontName="AU", fontSize=9, leading=12.5, spaceAfter=5)
st_li = ParagraphStyle("li", fontName="AU", fontSize=9, leading=12.5, leftIndent=12, spaceAfter=3)
st_cap = ParagraphStyle("cap", fontName="AU", fontSize=8, leading=10, textColor=colors.HexColor("#555555"),
                        spaceBefore=2, spaceAfter=8)
st_cell = ParagraphStyle("c", fontName="AU", fontSize=7.5, leading=9)
st_cell_h = ParagraphStyle("ch", fontName="AUB", fontSize=7.5, leading=9)

doc = SimpleDocTemplate("AVR2_report.pdf", pagesize=A4,
                        leftMargin=M, rightMargin=M, topMargin=M, bottomMargin=M,
                        title="AVR-2 Итоговое задание. Щетников Даниил", author="Даниил Щетников")

el = []


def P(text, style=st_p):
    el.append(Paragraph(text.replace("\n", "<br/>"), style))


def IMG(path, caption, width=None):
    from PIL import Image as PILImage
    import os
    if not os.path.exists(path):
        return
    w0, h0 = PILImage.open(path).size
    w = width or (W - 2 * M)
    el.append(Image(path, width=w, height=w * h0 / w0))
    el.append(Paragraph(caption, st_cap))


def TBL(header, rows):
    data = [[Paragraph(h, st_cell_h) for h in header]]
    for r in rows:
        data.append([Paragraph(str(c), st_cell) for c in r])
    t = Table(data, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    el.append(t)
    el.append(Spacer(1, 8))


P("AVR-2: Анализ временных рядов. Итоговое задание", st_title)
P("Студент: Щетников Даниил, группа BHEMAI-25-AVR-2. Репозиторий: "
  "github.com/Shchetnikovoff/AVR2-TimeSeries. Отчет версия 3 (доработка по замечаниям от 14.09.2026).")

P("1. Описание временного ряда", st_h1)
P("Суточный минимум температуры воздуха, Мельбурн (Австралия), 01.01.1981 - 31.12.1990. Источник: "
  "метеослужба Австралии BOM, файл daily-min-temperatures.csv из открытого набора Jason Brownlee. "
  "Характеристики: 3652 календарных дня, среднее 11.18 C, стандартное отклонение 4.07 C, минимум 0.0 C, "
  "максимум 26.3 C. Ряд с сильной годовой сезонностью (январь около 15.0 C, июль около 6.7 C). "
  "Практический смысл прогноза: нагрузка на отопление и охлаждение в энергетике, риск заморозков "
  "в сельском хозяйстве.")
IMG("img/01_series.png", "Рис. 1. Весь ряд, 1981-1990.")

P("2. Постановка задачи", st_h1)
P("Прогноз на h = 90 дней вперед по дневным данным. Разбиение фиксированное: train, первые 3562 дня "
  "(1981 - 02.10.1990), holdout, последние 90 дней (03.10.1990 - 31.12.1990). Метрики: RMSE, MAE, MAPE "
  "(значения по модулю меньше 0.5 C исключены из знаменателя), sMAPE. Надежность: rolling-бектест на 3 окнах "
  "по 90 дней со сдвигом 90, покрытие интервалов 80/95, анализ остатков (ACF + Ljung-Box), тест "
  "Диболда-Мариано на значимость различий между моделями.")

P("3. EDA", st_h1)
P("Пропуски: в исходном файле нет 2 календарных дат (1984 и 1988), ряд достроен до полного диапазона, "
  "пропуски заполнены линейной интерполяцией. Стационарность: тест Дики-Фуллера на уровнях, статистика "
  "-4.44, p-value 0.00025, ряд стационарен, d = 0. Сезонность: ACF(365) = 0.48, ACF(180) = -0.48. "
  "Аддитивная декомпозиция (период 365): сезонная компонента 0.81 дисперсии, тренд 0.01, остаток 0.18. "
  "PACF: лаг 1 доминирует (0.77), значимые лаги до 7, это обосновывает AR(7).")
IMG("img/02_seasonality.png", "Рис. 2. Сезонность: распределение по месяцам и среднегодовые значения.")
IMG("img/03_decomposition.png", "Рис. 3. Декомпозиция ряда, период 365.")
IMG("img/04_acf_pacf.png", "Рис. 4. ACF до 400 лагов и PACF до 30.")

P("4. Методы и обоснование параметров", st_h1)
P("Всего 14 конфигураций: 2 бейзлайна, 6 статистических (3 метода в ручном и авто режимах), 3 ML, 3 DL. "
  "Полный дамп параметров, seed и версий: results/model_configs.json. Seed 42 зафиксирован везде.")
TBL(["Группа", "Модель", "Ключевые параметры", "Обоснование"], [
    ["бейзлайн", "Naive", "последнее значение", "точка отсчета"],
    ["бейзлайн", "SeasonalNaive(365)", "лаг 365", "ACF(365)=0.48"],
    ["стат, ручной", "ARIMA(7,0,0)", "p=7, d=0, q=0", "d=0 по ADF, AR(7) по PACF"],
    ["стат, авто", "AutoARIMA(m=7)", "автоподбор", "сравнение с ручным"],
    ["стат, ручной", "HoltWinters", "m=365, аддитивные", "годовая сезонность"],
    ["стат, авто", "AutoETS(m=365)", "model=ZZZ", "автовыбор структуры"],
    ["стат, ручной", "Theta(m=365)", "классический", "сезонный ряд"],
    ["стат, авто", "AutoTheta(m=365)", "автоподбор", "сравнение с ручным"],
    ["ML", "RandomForest", "200 деревьев, seed 42", "бэггинг на лагах"],
    ["ML", "HistGradientBoosting", "default, seed 42", "бустинг"],
    ["ML", "Ridge", "alpha=1.0", "линейный контроль"],
    ["DL", "LSTM", "input 365, 500 шагов, seed 42", "годовой контекст"],
    ["DL", "MLP", "input 365, 500 шагов, seed 42", "полносвязная"],
    ["DL", "NHITS", "input 365, 500 шагов, seed 42", "иерархическая"],
])
P("Признаки для ML (mlforecast): лаги 1-7, 14, 21, 28, 35, 90, 180, 364, 365 плюс dayofweek, month, "
  "quarter, dayofyear. Набор лагей составлен из EDA: короткие по PACF, годовые по ACF.")

P("5. Результаты на holdout (90 дней)", st_h1)
TBL(["Метод", "RMSE", "MAE", "MAPE %", "sMAPE %", "Комментарий"], [
    ["DL-LSTM", "2.669", "1.986", "16.49", "15.81", "лучший на holdout, см. бектест"],
    ["HoltWinters", "2.721", "2.115", "17.17", "16.65", "лучший статметод"],
    ["Theta", "2.722", "2.115", "17.11", "16.65", "уровень HoltWinters"],
    ["AutoTheta", "2.722", "2.115", "17.11", "16.65", "сошелся к ручному"],
    ["ML-HGB", "2.770", "2.144", "17.60", "16.93", "лучший ML"],
    ["DL-MLP", "2.879", "2.106", "16.22", "16.76", "средний"],
    ["ML-Ridge", "2.948", "2.169", "16.31", "17.27", "стабильный контроль"],
    ["ML-RF", "2.986", "2.244", "17.32", "17.95", "бэггинг шумит"],
    ["DL-NHITS", "3.054", "2.270", "17.19", "18.39", "слабее LSTM и MLP"],
    ["AutoARIMA", "3.408", "2.688", "19.96", "21.59", "волну не держит"],
    ["ARIMA(7,0,0)", "3.417", "2.682", "19.89", "21.54", "ручной режим"],
    ["Naive", "3.724", "2.984", "21.94", "24.27", "точка отсчета"],
    ["SeasonalNaive", "3.750", "2.891", "23.07", "23.21", "годовой наив"],
    ["AutoETS", "3.799", "3.058", "22.40", "24.96", "автовыбор ошибся"],
])
P("Прогнозы всех 14 моделей на holdout, на каждой панели RMSE:", )
IMG("img/09_forecasts_all.png", "Рис. 5. Прогнозы всех 14 моделей на holdout против факта.")
IMG("img/05_forecasts_test.png", "Рис. 6. Лучшие представители семейств.")
IMG("img/06_metrics_bar.png", "Рис. 7. RMSE всех моделей на holdout.")

P("6. Анализ надежности", st_h1)
P("6.1. Бектест по окнам. Rolling-бектест, 3 окна по 90 дней (cutoff 05.04.1990, 04.07.1990, 02.10.1990).", )
TBL(["Модель", "Окно 1", "Окно 2", "Окно 3", "Среднее"], [
    ["ML-HGB", "2.406", "2.571", "2.770", "2.582"],
    ["DL-MLP", "2.433", "2.568", "2.797", "2.599"],
    ["ML-RF", "2.457", "2.538", "2.986", "2.660"],
    ["HoltWinters", "2.420", "2.851", "2.721", "2.664"],
    ["ML-Ridge", "2.694", "2.480", "2.948", "2.708"],
    ["Theta / AutoTheta", "2.518", "2.937", "2.722", "2.726"],
    ["DL-NHITS", "2.852", "2.711", "3.084", "2.882"],
    ["DL-LSTM", "3.475", "4.024", "2.568", "3.355"],
    ["ARIMA", "3.557", "3.114", "3.417", "3.363"],
    ["AutoARIMA", "3.590", "3.270", "3.408", "3.423"],
    ["SeasonalNaive", "3.381", "3.655", "3.750", "3.595"],
    ["AutoETS", "4.757", "2.638", "3.799", "3.731"],
    ["Naive", "5.420", "2.600", "3.724", "3.915"],
])
IMG("img/10_backtest_windows.png", "Рис. 8. RMSE по окнам бектеста.")
P("Ключевое наблюдение: DL-LSTM, лучший на единственном holdout (2.669), в бектесте девятый (3.355). "
  "Сеть выигрывает одно окно за счет гибкости, а не устойчивости. Лидеры бектеста: ML-HGB (2.582) и "
  "DL-MLP (2.599).")
P("6.2. Тест Диболда-Мариано (HAC, лаг 3). Отличие DL-LSTM от HoltWinters, Theta и ML-HGB статистически "
  "не значимо: p-value 0.67, 0.67, 0.61. Отличие от ARIMA (p=0.032), AutoARIMA (0.037), Naive (0.011), "
  "SeasonalNaive (0.0008), AutoETS (0.008) значимо. Формальная победа LSTM на holdout не означает "
  "реального превосходства над топ-группой.")
IMG("img/11_dm_heatmap.png", "Рис. 9. Значимость отличий от DL-LSTM на holdout.")
P("6.3. Вероятностные оценки. Интервалы HoltWinters: покрытие 80% интервала 0.811 при номинале 0.80, "
  "95% интервала 0.933 при номинале 0.95. Калибровка в пределах 0.02.")
IMG("img/07_intervals.png", "Рис. 10. Прогнозные интервалы HoltWinters на holdout.")
P("6.4. Остатки DL-LSTM на holdout: среднее 0.067 C (смещения нет), std 2.720, Ljung-Box(10) p=0.00041, "
  "в остатках осталась автокорреляция. Зафиксированный предел точности: RMSE 2.669 против std ряда 4.07.")
IMG("img/08_residuals_acf.png", "Рис. 11. Остатки лучшей модели и их ACF.")

P("7. Пайплайн и тестирование", st_h1)
P("pipeline.py: загрузка данных, предобработка, обучение HoltWinters(365) и SeasonalNaive(365), прогноз "
  "90 дней, метрики, замер времени. Прогон: HoltWinters RMSE=2.721 MAE=2.115 MAPE=17.17%, SeasonalNaive "
  "RMSE=3.750, время 2.5 c. Числа совпадают с ноутбуком до тысячных. Выбор HoltWinters обоснован разделом 6: "
  "результат статистически неотличим от лучшего (DM p=0.67), лучший статметод в бектесте (2.664), "
  "детерминированное обучение за секунды, интервалы откалиброваны.")

P("8. Верифицируемость", st_h1)
P("Все параметры моделей, seed и версии библиотек: results/model_configs.json. Воспроизведение: "
  "uv venv --python 3.12, uv pip install -r requirements.txt, затем ноутбук, report_figures.py, pipeline.py. "
  "Контроль детерминизма: повторный прогон воспроизводит метрики results/metrics_holdout.csv с точностью "
  "1e-9. ML: random_state=42, DL: random_seed=42. Все числа отчета берутся из results/*.csv.")

P("9. Выводы по задачам", st_h1)
P("Задача 1: ряд подготовлен (3652 дня, 2 даты интерполированы), ADF p=0.00025, сезонность 0.81 дисперсии, "
  "готовый ряд сохранен.")
P("Задача 2: лучшие статметоды HoltWinters 2.721 и Theta 2.722, лучше бейзлайнов более чем на градус. "
  "Ручные режимы не уступают авто. AutoETS 3.799 хуже наивного: автовыбор на сезонности 365 ненадежен.")
P("Задача 3: лучший ML HGB (2.770 holdout, 2.582 бектест). Лучший DL на holdout LSTM 2.669, но 3.355 "
  "в бектесте. DM-тест: LSTM неотличим от HoltWinters, Theta, HGB (p>0.6), значимо лучше наивов и ARIMA "
  "(p<0.04).")
P("Задача 4: пайплайн воспроизводит числа ноутбука за 2.5 c, отчет содержит все разделы чек-листа, "
  "все числа из результатов прогона.")

P("10. Общее заключение", st_h1)
P("Данные: суточный минимум температуры в Мельбурне, 1981-1990, 3652 дня. Цель: прогноз на 90 дней и "
  "обоснованный выбор метода среди 14 конфигураций (2 бейзлайна, 6 статистических, 3 ML, 3 DL). Итоги: "
  "лучший на holdout DL-LSTM RMSE 2.669 C (MAE 1.986, MAPE 16.49%). Самый устойчивый в бектесте ML-HGB "
  "2.582. Лучший статистический HoltWinters 2.721 (holdout) и 2.664 (бектест). Бейзлайны 3.724 и 3.750. "
  "DM-тест: различие LSTM с топ-группой не значимо (p>0.6), уверенно только превосходство над наивами и "
  "ARIMA (p<0.04). Интервалы HoltWinters откалиброваны (0.811 и 0.933 при номиналах 0.80 и 0.95). "
  "В остатках лучшей модели сохраняется автокорреляция (Ljung-Box p=0.00041). Практический вывод: "
  "в эксплуатацию идут HoltWinters или ML-HGB, LSTM отложен как нестабильный. Пайплайн воспроизводит "
  "результат за 2.5 c, конфигурации и версии открыты в results/model_configs.json.")

doc.build(el)
import os
print("ok", os.path.getsize("AVR2_report.pdf") // 1024, "KB")
