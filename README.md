# Dice calculator

This is a simple dice distribution calculator. Input an expression like `3d6 + 2` and it will output the distribution of the sum of the dice rolls.

Expressions can get arbitrarily complex, with repetition, arithmetic operators, and functions.

For example, to compute a roll of 2d6 with advantage, you can use `3kh2d6` - read as "keep the highest 2 of 3 dice rolled".

To take the maximum of two results, you can use the `^` operator, or the `max` function (which can take any number of arguments).
