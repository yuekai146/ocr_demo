import inspect
from wtforms import Form, FloatField, TextField, IntegerField, validators
from compute import compute


args = inspect.getfullargspec(compute)[0]
defaults = inspect.getfullargspec(compute)[3]
defaults = [None] * (len(args)  - len(defaults)) + list(defaults)


type2form = {
        type(1): IntegerField,
        type(1.0): FloatField,
        type(''): TextField
        }


class InputForm(Form):
    pass


for arg, value in zip(args, defaults):
    if value is None:
        setattr(InputForm, arg, FloatField(validators=[validators.InputRequired()]))
    else:
        if type(value) in type2form:
            field = type2form[type(value)]
            setattr(InputForm, arg, field(default=value, validators=[validators.InputRequired()]))
        else:
            raise TypeError('Argument {}:{} not supported'.format(arg, value))
