Install :

pacman -S octave

Run :

octave

Add current dir to path ( to be able to run stuff ) :

addpath ("~/scripts/octave")

I'm pretty sure as long as the parent directory is in the path it should compile
and run files in subdirectories

To load / execute a file , make sure you are in the right dir with pwd :

myFile

where myFile is just the filename without the extension. There are somefile
types that might use the extension , but right now I am only using .m files.

i also tried to specify the explicit path , it didnt seem to work

RCFILE :

~/.octaverc

https://octave.org/doc/v5.2.0/Startup-Files.html

CLEARING THE SCREEN :

clc

Meta-D: clear the next word1
Ctrl-K: clear to the end of the line
Ctrl-U: clear the whole line
Ctrl-L: clear the line and the screen

CHANGING THE DEFAULT PROMPT :

PS1('>> ')

IF THE ONLY THING IN THE FILE IS A FUNCTION
and you are getting the undefined variable error then:
https://stackoverflow.com/questions/44508581/octave-gnu-undefined-variable-x-even-though-its-defined-as-function-input


ADD FOLLOWING TO RIFLE.CONF TO OPEN FROM RANGER

Basically just add the m option to the end of the following two sections

#-------------------------------------------
# Misc
#-------------------------------------------
!mime ^text, label editor, ext xml|json|csv|tex|py|pl|rb|js|sh|php|m = ${VISUAL:-$EDITOR} -- "$@"
!mime ^text, label pager,  ext xml|json|csv|tex|py|pl|rb|js|sh|php|m = "$PAGER" -- "$@"

#--------------------------------------------
# Scripts
#-------------------------------------------
ext m   = octave -- "$1"


