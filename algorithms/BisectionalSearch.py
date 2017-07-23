# Bisectional Search:
"""
The program works as follows: you (the user) thinks of an integer between 0 (inclusive) and 100 (not inclusive). 
The computer makes guesses, and you give it input - is its guess too high or too low? Using bisection search, the computer will guess the user's secret number!
"""
import math;

print('Please think of a number between 0 and 100!');

h = 100;
l = 0;
ans = (h+l)//2;

while True:
    print('Is your secret number '+str(ans)+'?')
    inp= input('Enter \'h\' to indicate the guess is too high. Enter \'l\' to indicate the guess is too low. Enter \'c\' to indicate I guessed correctly.');
    
    if inp == 'h':
        h = ans;
        ans = (h+l)//2;
    elif inp == 'l':
        l = ans;
        ans = (h+l)//2;
    elif inp == 'c':
        break
    elif inp!= 'h' or inp!= 'l' or inp!= 'c':
        print('Sorry, I did not understand your input.');
 
print('Game over. Your secret number was: '+str(ans));
