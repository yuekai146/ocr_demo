from flask import Flask, request, render_template
from compute import compute
from inputs import InputForm
import inspect


app = Flask(__name__)


@app.route('/uy_ocr', methods=['POST', 'GET'])
def uy_ocr():
    args = inspect.getfullargspec(compute)[0]
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        kwargs = {arg:getattr(form, arg).data for arg in args if hasattr(form, arg)}
        print(kwargs)
        result = compute(**kwargs)
    else:
        result = None

    return render_template('view.html', form=form, result=result)
