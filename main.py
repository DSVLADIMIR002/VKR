import pandas as pd
import joblib
from keras.models import load_model
from flask import Flask, render_template, request, flash
from flask_wtf import FlaskForm
from wtforms import IntegerField, FloatField, SubmitField
from wtforms.validators import DataRequired, Length
class Model_predict(FlaskForm):
    density = FloatField('Плотность, кг/м3  ', validators=[DataRequired()])
    modulus_of_elasticity = FloatField('модуль упругости, ГПа ', validators=[DataRequired()])
    amount_of_hardener = FloatField(' Количество отвердителя ', validators=[DataRequired()])
    content_of_epoxy_groups = FloatField('Содержание эпоксидных групп', validators=[DataRequired()])
    flash_point = FloatField('Температура вспышки, С_2 ', validators=[DataRequired()])
    surface_density = FloatField(' Поверхностная плотность, г/м2', validators=[DataRequired()])
    tensile_modulus = FloatField('Модуль упругости при растяжении, ГПа', validators=[DataRequired()])
    tensile_strength = FloatField('Прочность при растяжении, МПа ', validators=[DataRequired()])
    resin_consumption = FloatField('Потребление смолы, г/м2', validators=[DataRequired()])
    patch_angle = IntegerField('Угол нашивки, град ', validators=[DataRequired()])
    patch_pitch = FloatField('Шаг нашивки ', validators=[DataRequired()])
    patch_density = FloatField('Плотность нашивки ', validators=[DataRequired()])
    submit = SubmitField('Расчитать')


app = Flask(__name__)
app.secret_key = 'your_secret_key'
scaler = joblib.load('scaler.pkl')
model = load_model('model.keras')
@app.route("/", methods=["GET", "POST"])
def index():
    form = Model_predict()
    if request.method == 'GET':
        return render_template("index.html", form=form)
    if form.validate_on_submit():
        density = form.density.data
        modulus_of_elasticity = form.modulus_of_elasticity.data
        amount_of_hardener = form.amount_of_hardener.data
        content_of_epoxy_groups = form.content_of_epoxy_groups.data
        flash_point = form.flash_point.data
        surface_density = form.surface_density.data
        tensile_modulus = form.tensile_modulus.data
        tensile_strength = form.tensile_strength.data
        resin_consumption = form.resin_consumption.data
        patch_angle = form.patch_angle.data
        patch_pitch = form.patch_pitch.data
        patch_density = form.patch_density.data
        df = pd.DataFrame(data= [[density,
                                  modulus_of_elasticity,
                                  amount_of_hardener,
                                  content_of_epoxy_groups,
                                  flash_point,
                                  surface_density,
                                  tensile_modulus,
                                  tensile_strength,
                                  resin_consumption,
                                  patch_angle,
                                  patch_pitch,
                                  patch_density,  ]], columns= ['Плотность, кг/м3',
       'модуль упругости, ГПа', 'Количество отвердителя, м.%',
       'Содержание эпоксидных групп,%_2', 'Температура вспышки, С_2',
       'Поверхностная плотность, г/м2', 'Модуль упругости при растяжении, ГПа',
       'Прочность при растяжении, МПа', 'Потребление смолы, г/м2',
       'Угол нашивки, град', 'Шаг нашивки', 'Плотность нашивки'])
        df['angle_90'] = df['Угол нашивки, град'].apply(lambda x: 1 if x == 90 else 0)
        df = df.drop('Угол нашивки, град', axis=1)
        x = scaler.transform(df)
        y = model.predict(x)
        return render_template("result.html", y = y[0][0] )
if __name__ == '__main__':
     app.run(debug=True, host='0.0.0.0', port= 5000)