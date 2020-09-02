cyr 
===
cyr is a simple console program that converts text written in plain ASCII to UTF-8 cyrillic.

About
-----
* Authors:     see the `LICENSE`
* License:     MIT License
* Bug reports: https://github.com/Aleksej10/cyrillic/issues
* git clone    https://github.com/Aleksej10/cyrillic

Dependencies
------------
* python (>=3.6)
* torch 
* numpy
* `nohup`

Installing
----------
Install through PyPI: 
```
pip install cyr
```
Download `.cyr_nets` and move them to your home directory (`mv .cyr_nets ~/`)

Example use
-----------
If you know you're going to use it multiple times in a short period of time, it is best to use
it as a daemon to avoid loading nets (which takes most of the time) every time you run it.   
Start daemon:
``` 
cyr -D 
```
Convert `file` in place using daemon:
```
cyr -d -f file -i
```
Kill daemon when you're done:
```
cyr -K
```
In case of error check `~/.cyr_nohup` for logs.
Check the `man` page after installing for more options.




