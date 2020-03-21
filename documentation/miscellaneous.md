# Miscellaneous
  
Here we present a set of different things that we found useful in the past

* ## [Checksums](#Checksums_P)
* ## [Emacs](#Emacs_P)


## <a id="Checksums_P"></a> Checksums

When transfering large amounts of data, or very important data, among machines, it is important to verify that the integrity of the data transfered. Checksums can be used for this. Say you have a folder that want to transfer from San Diego to Princeton. The way to do that is:

```sh
cd my_folder/
find -type f \! -name SHA224SUMS -exec sha224sum \{\} \+ > SHA224SUMS
```

The above command will create a file called SHA224SUMS with the checksums of all files in that folder. Once that folder has been transfered to another machine, the integrity of the data can be checked by executing the following command:

```sh
sha224sum -c SHA224SUMS --quiet
```

If nothing is printed out, the data has been properly transfered.


## <a id="Emacs_P"></a> Emacs

Sometimes, when using a new machine, the emacs files do not have the dimensions we want, e.g. the do not fit well in the screen. This can be fixed by specifying directly the geometry when calling emacs

```sh
emacs --geometry=88x37 library.py
```

Thus, once the geometry that best fit the screen is found, it is enough to add this line to the ~/.bashrc file:

```sh
alias emacs="emacs --geometry=88x37"
```