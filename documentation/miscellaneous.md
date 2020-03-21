# Miscellaneous
  
Here we present a set of different things that we found useful in the past

* ## [Emacs](#Emacs)


## <a id="Emacs"></a> Emacs

Sometimes, when using a new machine, the emacs files do not have the dimensions we want, e.g. the do not fit well in the screen. This can be fixed by specifying directly the geometry when calling emacs

```sh
emacs --geometry=88x37 library.py
```

Thus, once the geometry that best fit the screen is found, it is enough to add this line to the ~/.bashrc file:

```sh
alias emacs="emacs --geometry=88x37"
```