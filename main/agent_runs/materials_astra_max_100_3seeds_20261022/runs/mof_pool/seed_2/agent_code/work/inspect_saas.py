import inspect
import botorch.fit as bf
import botorch.models.map_saas as ms
for module in [bf, ms]:
    for name in dir(module):
        if 'saas' in name.lower() and ('fit' in name.lower() or 'single' in name.lower()):
            obj = getattr(module,name)
            if callable(obj):
                print(name, inspect.signature(obj))
                print(inspect.getdoc(obj))
