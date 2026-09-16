"""Qt bindings when Qt is installed, and small stand-ins when it is not.

The README calls this a library "independent of any radio hardware or GUI", and
that was true of everything except the imports: every module began with
`from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot`, so `import sdr_core`
failed outright on a machine with no GUI toolkit — a CI runner, a headless box,
anyone who wanted the DSP and not the interface.

Qt is used here for exactly three things: a base class, signals that carry
results to whoever is listening, and a decorator that marks a method as a slot.
None of that needs a windowing system, so when PyQt6 is absent these plain-Python
equivalents take over. The API is the same either way — `connect`, `emit`, and a
`pyqtSlot` decorator that passes the function through — so code written against
Qt keeps working, and so does code that never installs it.
"""

try:  # pragma: no cover - exercised only where PyQt6 is installed
    from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot  # noqa: F401

except ImportError:  # pragma: no cover - the path CI and headless users take

    class _BoundSignal:
        """One signal on one object: a list of callbacks and a way to fire them."""

        def __init__(self):
            self._subscribers = []

        def connect(self, callback):
            self._subscribers.append(callback)

        def disconnect(self, callback=None):
            if callback is None:
                self._subscribers.clear()
            elif callback in self._subscribers:
                self._subscribers.remove(callback)

        def emit(self, *args):
            # A listener that raises must not stop the others, which is also
            # Qt's behaviour for queued connections.
            for callback in list(self._subscribers):
                callback(*args)

    class pyqtSignal:
        """Class attribute that becomes a per-instance signal on first access."""

        def __init__(self, *types):
            self.types = types
            self._attr = None

        def __set_name__(self, owner, name):
            self._attr = f"_signal_{name}"

        def __get__(self, instance, owner=None):
            if instance is None:
                return self
            bound = instance.__dict__.get(self._attr)
            if bound is None:
                bound = _BoundSignal()
                instance.__dict__[self._attr] = bound
            return bound

    def pyqtSlot(*args, **kwargs):
        """Marks a method as a slot. Without Qt there is nothing to register."""

        def decorate(function):
            return function

        return decorate

    class QObject:
        """Enough of QObject for classes that only inherit it to own signals."""

        def __init__(self, parent=None):
            self._parent = parent

        def parent(self):
            return self._parent
