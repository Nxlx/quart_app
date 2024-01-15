
from datetime import datetime
from flask import Flask, redirect, render_template, url_for, request 
from flask_bootstrap import Bootstrap4
from flask_sqlalchemy import SQLAlchemy
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, SelectMultipleField
from wtforms.validators import DataRequired


app = Flask(__name__)
bootstrap = Bootstrap4(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///events.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['WTF_CSRF_ENABLED'] = False  
db = SQLAlchemy(app)
admin = Admin(app, name='microblog', template_mode='bootstrap3')

# Определение модели бд для событий
class Event(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    name = db.Column(db.String(255), nullable=False)
    duration = db.Column(db.Integer, nullable=False)

    def __str__(self):
        return (
            f"Название: {self.name}\n"
            f"Дата: {self.date}\n"
            f"Продолжительность {self.duration}ч"
        )

# Определение модели бд для частей
class Part1(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    part_number = db.Column(db.Integer, nullable=False)
    value = db.Column(db.String(255), nullable=False)

class Part2(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    part_number = db.Column(db.Integer, nullable=False)
    value = db.Column(db.String(255), nullable=False)

class Part3(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    part_number = db.Column(db.Integer, nullable=False)
    value = db.Column(db.String(255), nullable=False) 

class Part4(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    part_number = db.Column(db.Integer, nullable=False)
    value = db.Column(db.String(255), nullable=False)
class Part5(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    part_number = db.Column(db.Integer, nullable=False)
    value = db.Column(db.String(255), nullable=False)

class Part6(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    part_number = db.Column(db.Integer, nullable=False)
    value = db.Column(db.String(255), nullable=False)

class Part7(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    part_number = db.Column(db.Integer, nullable=False)
    value = db.Column(db.String(255), nullable=False) 

class Part8(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    part_number = db.Column(db.Integer, nullable=False)
    value = db.Column(db.String(255), nullable=False)
class Part9(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    part_number = db.Column(db.Integer, nullable=False)
    value = db.Column(db.String(255), nullable=False)

class Part10(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    part_number = db.Column(db.Integer, nullable=False)
    value = db.Column(db.String(255), nullable=False)

class Part11(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    part_number = db.Column(db.Integer, nullable=False)
    value = db.Column(db.String(255), nullable=False) 

class Part12(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    part_number = db.Column(db.Integer, nullable=False)
    value = db.Column(db.String(255), nullable=False)
class Part13(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    part_number = db.Column(db.Integer, nullable=False)
    value = db.Column(db.String(255), nullable=False)

class Part14(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    part_number = db.Column(db.Integer, nullable=False)
    value = db.Column(db.String(255), nullable=False)

# Представление модели для событий
class EventView(ModelView):
    column_display_pk = True  
    form_columns = ['date', 'name', 'duration']

# Представление модели для частей
class PartsView(ModelView):
    column_display_pk = True
    form_columns = ['part_number', 'value']

    def add_view(self, admin, model, session):
        for part in self.parts:
            admin.add_view(ModelView(part, session, category=model.__name__))

# Регистрация представлений моделей в административной панели
admin.add_view(EventView(Event, db.session))
admin.add_view(PartsView(Part1, db.session, category='Parts'))
admin.add_view(PartsView(Part2, db.session, category='Parts'))
admin.add_view(PartsView(Part3, db.session, category='Parts'))
admin.add_view(PartsView(Part4, db.session, category='Parts'))
admin.add_view(PartsView(Part5, db.session, category='Parts'))
admin.add_view(PartsView(Part6, db.session, category='Parts'))
admin.add_view(PartsView(Part7, db.session, category='Parts'))
admin.add_view(PartsView(Part8, db.session, category='Parts'))
admin.add_view(PartsView(Part9, db.session, category='Parts'))
admin.add_view(PartsView(Part10, db.session, category='Parts'))
admin.add_view(PartsView(Part11, db.session, category='Parts'))
admin.add_view(PartsView(Part12, db.session, category='Parts'))
admin.add_view(PartsView(Part13, db.session, category='Parts'))
admin.add_view(PartsView(Part14, db.session, category='Parts'))

# Определение модели бд для новой таблицы
class MassData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    value = db.Column(db.String(255), nullable=False)
    part_number = db.Column(db.Integer, nullable=False)

# Определение формы для массового добавления данных
class MassAddForm(FlaskForm):
    data = StringField('Данные', validators=[DataRequired()])
    part_number = SelectField('Выберите часть', choices=[(str(i), f'Часть {i}') for i in range(1, 15)], coerce=int, validators=[DataRequired()])


# Маршрут для просмотра массовых данных
@app.route('/view_mass_data')
def view_mass_data():
    mass_data = MassData.query.all()
    return render_template("view_mass_data.html", h1="Массовые данные", mass_data=mass_data)

# Определение формы для добавления частей
class AddPartForm(FlaskForm):
    value = StringField('Значение', validators=[DataRequired()])
    part_number = SelectField('Выберите часть', choices=[(str(i), f'Часть {i}') for i in range(1, 15)], coerce=int, validators=[DataRequired()])
    #value = StringField('Значение', validators=[DataRequired()])
    #part_number = SelectField('Выберите часть', choices=[(str(i), f'Часть {i}') for i in range(1, 15)], coerce=int, validators=[DataRequired()])
    #part_number = SelectMultipleField('Выберите часть', choices=[(str(i), f'Часть {i}') for i in range(1, 15)], coerce=int, validators=[DataRequired()])

def handle_selected_parts(selected_parts, action):
    # Обработка различных действий
    if action == 'delete':
        # Удаление из базы данных
        for part_id in selected_parts:
            part = None

            part1 = Part1.query.get(part_id)
            part2 = Part2.query.get(part_id)
            part3 = Part3.query.get(part_id)
            part4 = Part4.query.get(part_id)
            part5 = Part5.query.get(part_id)
            part6 = Part6.query.get(part_id)
            part7 = Part7.query.get(part_id)
            part8 = Part8.query.get(part_id)
            part9 = Part9.query.get(part_id)
            part10 = Part10.query.get(part_id)
            part11 = Part11.query.get(part_id)
            part12 = Part12.query.get(part_id)
            part13 = Part13.query.get(part_id)
            part14 = Part14.query.get(part_id)

            if part1:
                part = part1
            elif part2:
                part = part2
            elif part3:
                part = part3
            elif part4:
                part = part4
            elif part5:
                part = part5
            elif part6:
                part = part6
            elif part7:
                part = part7
            elif part8:
                part = part8
            elif part9:
                part = part9
            elif part10:
                part = part10
            elif part11:
                part = part11
            elif part12:
                part = part12
            elif part13:
                part = part13
            elif part14:
                part = part14
            if part:
                db.session.delete(part)
                db.session.commit()


# Маршрут для просмотра событий
@app.route("/events")
def view_events():
    events = Event.query.order_by(Event.date).all()
    return render_template("events.html", h1="События", events=events)

# Маршрут для просмотра частей с объединенными данными
@app.route("/parts_combined")
def view_parts_combined():
    parts1 = Part1.query.all()
    parts2 = Part2.query.all()
    parts3 = Part3.query.all()
    parts4 = Part4.query.all()
    parts5 = Part5.query.all()
    parts6 = Part6.query.all()
    parts7 = Part7.query.all()
    parts8 = Part8.query.all()
    parts9 = Part9.query.all()
    parts10 = Part10.query.all()
    parts11 = Part11.query.all()
    parts12 = Part12.query.all()
    parts13 = Part13.query.all()
    parts14 = Part14.query.all()

    # В представлении Flask
    parts_combined = []

    # Создаем матрицу из пустых значений
    max_len = max(len(parts1), len(parts2), len(parts3), len(parts4), len(parts5), len(parts6), len(parts7), len(parts8),len(parts9), len(parts10), len(parts11), len(parts12), len(parts13), len(parts14))
    parts_combined = [
        {'part1': None, 'part2': None, 'part3': None, 'part4': None, 'part5': None, 'part6': None, 'part7': None, 'part8': None, 'part9': None, 'part10': None, 'part11': None, 'part12': None, 'part13': None, 'part14': None}
        for _ in range(max_len)
    ]

    # Заполняем матрицу значениями из Part1
    for i, part1 in enumerate(parts1):
        parts_combined[i]['part1'] = part1.value

    # Заполняем матрицу значениями из Part2
    for i, part2 in enumerate(parts2):
        parts_combined[i]['part2'] = part2.value

    # Заполняем матрицу значениями из Part3
    for i, part3 in enumerate(parts3):
        parts_combined[i]['part3'] = part3.value

    # Заполняем матрицу значениями из Part4
    for i, part4 in enumerate(parts4):
        parts_combined[i]['part4'] = part4.value
    
    # Заполняем матрицу значениями из Part5
    for i, part5 in enumerate(parts5):
        parts_combined[i]['part5'] = part5.value

    # Заполняем матрицу значениями из Part6
    for i, part6 in enumerate(parts6):
        parts_combined[i]['part6'] = part6.value

    # Заполняем матрицу значениями из Part7
    for i, part7 in enumerate(parts7):
        parts_combined[i]['part7'] = part7.value

    # Заполняем матрицу значениями из Part8
    for i, part8 in enumerate(parts8):
        parts_combined[i]['part8'] = part8.value

        # Заполняем матрицу значениями из Part9
    for i, part9 in enumerate(parts9):
        parts_combined[i]['part9'] = part9.value

    # Заполняем матрицу значениями из Part10
    for i, part10 in enumerate(parts10):
        parts_combined[i]['part10'] = part10.value
    
    # Заполняем матрицу значениями из Part11
    for i, part11 in enumerate(parts11):
        parts_combined[i]['part11'] = part11.value

    # Заполняем матрицу значениями из Part12
    for i, part12 in enumerate(parts12):
        parts_combined[i]['part12'] = part12.value

    # Заполняем матрицу значениями из Part13
    for i, part13 in enumerate(parts13):
        parts_combined[i]['part13'] = part13.value

    # Заполняем матрицу значениями из Part14
    for i, part14 in enumerate(parts14):
        parts_combined[i]['part14'] = part14.value

    parts_combined_sorted = [{key: value for key, value in row.items() if value is not None} for row in parts_combined]
    return render_template("parts_combined.html", h1="Части (Объединенные)", parts_combined=parts_combined_sorted)



# Маршрут для добавления событий
@app.route('/', methods=['POST'])
def add_event():
    date = datetime.strptime(request.form['eventDate'], '%Y-%m-%d').date()
    name = request.form['eventName']
    duration = int(request.form['eventDuration'])
    event = Event(date=date, name=name, duration=duration)
    db.session.add(event)
    db.session.commit()
    return redirect('/')

# Маршрут для главной страницы
@app.route("/")
def index():
    return render_template("index.html", h1="Главная страница")

# Маршрут для страницы "О приложении"
@app.route("/about")
def get_page_about():
    return render_template("about.html", h1="О приложении")

# Маршрут для просмотра частей
@app.route("/parts")
def view_parts2():
    parts1 = Part1.query.all()
    parts2 = Part2.query.all()
    parts3 = Part3.query.all()
    parts4 = Part4.query.all()
    parts5 = Part5.query.all()
    parts6 = Part6.query.all()
    parts7 = Part7.query.all()
    parts8 = Part8.query.all()
    parts9 = Part9.query.all()
    parts10 = Part10.query.all()
    parts11 = Part11.query.all()
    parts12 = Part12.query.all()
    parts13 = Part13.query.all()
    parts14 = Part14.query.all()

    return render_template("parts.html", h1="Части", parts1=parts1, parts2=parts2, parts3=parts3, parts4=parts4, parts5=parts5, parts6=parts6, parts7=parts7, parts8=parts8, parts9=parts9, parts10=parts10, parts11=parts11, parts12=parts12, parts13=parts13, parts14=parts14)

# Маршрут и представление для добавления частей
@app.route('/add_part', methods=['POST'])
def add_part():
    form = AddPartForm()

    if form.validate_on_submit():
        value = form.value.data  # Одно значение
        part_number = form.part_number.data  # Номер части

        if part_number == 1:
            part = Part1(value=value, part_number=1)
        elif part_number == 2:
            part = Part2(value=value, part_number=2)
        elif part_number == 3:
            part = Part3(value=value, part_number=3)
        elif part_number == 4:
            part = Part4(value=value, part_number=4)
        elif part_number == 5:
            part = Part5(value=value, part_number=5)
        elif part_number == 6:
            part = Part6(value=value, part_number=6)
        elif part_number == 7:
            part = Part7(value=value, part_number=7)
        elif part_number == 8:
            part = Part8(value=value, part_number=8)
        elif part_number == 9:
            part = Part9(value=value, part_number=9)
        elif part_number == 10:
            part = Part10(value=value, part_number=10)
        elif part_number == 11:
            part = Part11(value=value, part_number=11)
        elif part_number == 12:
            part = Part12(value=value, part_number=12)
        elif part_number == 13:
            part = Part13(value=value, part_number=13)
        elif part_number == 14:
            part = Part14(value=value, part_number=14)
        
        else:
            return render_template("error.html", h1="Ошибка", message="Неверный номер части")

        db.session.add(part)
        db.session.commit()

        return redirect(url_for('view_parts_combined'))

    return render_template("add_part.html", h1="ДОБАВИТЬ ЧАСТЬ", form=form)

# Маршрут и представление для массового добавления данных
@app.route('/mass_add_data', methods=['GET', 'POST'])
def mass_add_data():
    form = MassAddForm()

    if form.validate_on_submit():
        data = form.data.data
        part_number = form.part_number.data

        # Разделение данных по запятой
        data_blocks = [block.strip() for block in data.split(',')]

        for data_block in data_blocks:
#            mass_data = MassData(value=data_block, part_number=part_number)
            if part_number == 1:
                mass_data_block = Part1(value=data_block, part_number=1)
            elif part_number == 2:
                mass_data_block = Part2(value=data_block, part_number=2)
            elif part_number == 3:
                mass_data_block = Part3(value=data_block, part_number=3)
            elif part_number == 4:
                mass_data_block = Part4(value=data_block, part_number=4)
            elif part_number == 5:
                mass_data_block = Part5(value=data_block, part_number=5)
            elif part_number == 6:
                mass_data_block = Part6(value=data_block, part_number=6)
            elif part_number == 7:
                mass_data_block = Part7(value=data_block, part_number=7)
            elif part_number == 8:
                mass_data_block = Part8(value=data_block, part_number=8)
            elif part_number == 9:
                mass_data_block = Part9(value=data_block, part_number=9)
            elif part_number == 10:
                mass_data_block = Part10(value=data_block, part_number=10)
            elif part_number == 11:
                mass_data_block = Part11(value=data_block, part_number=11)
            elif part_number == 12:
                mass_data_block = Part12(value=data_block, part_number=12)
            elif part_number == 13:
                mass_data_block = Part13(value=data_block, part_number=13)
            elif part_number == 14:
                mass_data_block = Part14(value=data_block, part_number=14)
            else:
                return render_template("error.html", h1="Ошибка", message="Неверный номер части")           
            
            
            db.session.add(mass_data_block)

            db.session.commit()

        return redirect(url_for('mass_add_data'))

    return render_template("mass_add_data.html", h1="МАССОВОЕ ДОБАВЛЕНИЕ ДАННЫХ", form=form)


# Маршрут для просмотра формы добавления частей
@app.route('/add_part_view')
def add_part_view():
    form = AddPartForm()  # Создаем экземпляр формы
    return render_template("add_part.html", h1="ДОБАВИТЬ ЧАСТЬ", form=form)

# Маршрут для обработки выбора пользователя
@app.route('/process_selection', methods=['POST'])
def process_selection():
    selected_parts = request.form.getlist('selected_parts')
    action = request.form.get('action')

    handle_selected_parts(selected_parts, action)

    return redirect(url_for('view_parts2'))

# Обработчик ошибки 404
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', message="Page not found"), 404

# Запуск приложения Flask
if __name__ == "__main__":
   with app.app_context():
       db.create_all()
   app.run(debug=True)
