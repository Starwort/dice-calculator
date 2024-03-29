use std::cmp::Reverse;
use std::collections::HashMap;
use std::iter::Sum;
use std::num::TryFromIntError;
use std::ops::{
    Add,
    AddAssign,
    BitOr,
    BitOrAssign,
    Mul,
    MulAssign,
    Neg,
    Sub,
    SubAssign,
};

use itertools::Itertools;
use num::pow::Pow;
use num::rational::Ratio;
use num::{range, BigInt, BigUint, Integer, Signed, ToPrimitive, Zero};

pub type Value = Ratio<BigInt>;
pub type Count = BigUint;

#[inline]
fn uint(val: usize) -> BigUint {
    BigUint::from(val)
}

#[derive(Debug, Clone)]
pub struct Distribution {
    pub contents: HashMap<Value, Count>,
    pub count: Count,
}
impl From<isize> for Distribution {
    fn from(value: isize) -> Self {
        Self::new(
            [(Ratio::new(value.into(), 1.into()), uint(1))].into(),
            uint(1),
        )
    }
}
impl From<Value> for Distribution {
    fn from(value: Value) -> Self {
        Self::new([(value, uint(1))].into(), uint(1))
    }
}
impl Sum for Distribution {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::new(HashMap::new(), Count::zero()), Add::add)
    }
}
impl Add<Distribution> for Distribution {
    type Output = Distribution;

    fn add(mut self, rhs: Distribution) -> Self::Output {
        self += rhs;
        self
    }
}
impl Add<&Distribution> for Distribution {
    type Output = Distribution;

    fn add(mut self, rhs: &Distribution) -> Self::Output {
        self += rhs;
        self
    }
}
impl Add<Distribution> for &Distribution {
    type Output = Distribution;

    fn add(self, rhs: Distribution) -> Self::Output {
        rhs + self
    }
}
impl Add<&Distribution> for &Distribution {
    type Output = Distribution;

    fn add(self, rhs: &Distribution) -> Self::Output {
        let mut result = HashMap::new();
        if self.is_empty() || rhs.is_empty() {
            return Distribution::empty();
        }
        for (lhs, lhs_count) in &self.contents {
            for (rhs, rhs_count) in &rhs.contents {
                *result.entry(lhs + rhs).or_insert(Count::zero()) +=
                    lhs_count * rhs_count;
            }
        }
        Distribution::new(result, &self.count * &rhs.count)
    }
}
impl AddAssign<Distribution> for Distribution {
    fn add_assign(&mut self, rhs: Distribution) {
        *self += &rhs;
    }
}
impl AddAssign<&Distribution> for Distribution {
    fn add_assign(&mut self, rhs: &Distribution) {
        if self.is_empty() || rhs.is_empty() {
            *self = Self::empty();
            return;
        }
        let mut result = HashMap::new();
        for (lhs, lhs_count) in &self.contents {
            for (rhs, rhs_count) in &rhs.contents {
                *result.entry(lhs + rhs).or_insert(Count::zero()) +=
                    lhs_count * rhs_count;
            }
        }
        self.contents = result;
        self.count *= &rhs.count;
    }
}
impl Sub<Distribution> for Distribution {
    type Output = Distribution;

    fn sub(self, rhs: Distribution) -> Self::Output {
        self - &rhs
    }
}
impl Sub<Distribution> for &Distribution {
    type Output = Distribution;

    fn sub(self, rhs: Distribution) -> Self::Output {
        self - &rhs
    }
}
impl Sub<&Distribution> for Distribution {
    type Output = Distribution;

    fn sub(mut self, rhs: &Distribution) -> Self::Output {
        self -= rhs;
        self
    }
}
impl SubAssign<Distribution> for Distribution {
    fn sub_assign(&mut self, rhs: Distribution) {
        *self -= &rhs;
    }
}
impl SubAssign<&Distribution> for Distribution {
    fn sub_assign(&mut self, rhs: &Distribution) {
        if self.is_empty() || rhs.is_empty() {
            *self = Self::empty();
            return;
        }
        let mut result = HashMap::new();
        for (lhs, lhs_count) in &self.contents {
            for (rhs, rhs_count) in &rhs.contents {
                *result.entry(lhs - rhs).or_insert(Count::zero()) +=
                    lhs_count * rhs_count;
            }
        }
        self.contents = result;
        self.count *= &rhs.count;
    }
}
impl Sub<&Distribution> for &Distribution {
    type Output = Distribution;

    fn sub(self, rhs: &Distribution) -> Self::Output {
        if self.is_empty() || rhs.is_empty() {
            return Distribution::empty();
        }
        let mut result = HashMap::new();
        for (lhs, lhs_count) in &self.contents {
            for (rhs, rhs_count) in &rhs.contents {
                *result.entry(lhs - rhs).or_insert(Count::zero()) +=
                    lhs_count * rhs_count;
            }
        }
        Distribution::new(result, &self.count * &rhs.count)
    }
}
impl Mul<Distribution> for Distribution {
    type Output = Distribution;

    fn mul(mut self, rhs: Distribution) -> Self::Output {
        self *= &rhs;
        self
    }
}
impl Mul<&Distribution> for Distribution {
    type Output = Distribution;

    fn mul(mut self, rhs: &Distribution) -> Self::Output {
        self *= rhs;
        self
    }
}
impl MulAssign<Distribution> for Distribution {
    fn mul_assign(&mut self, rhs: Distribution) {
        *self *= &rhs;
    }
}
impl MulAssign<&Distribution> for Distribution {
    fn mul_assign(&mut self, rhs: &Distribution) {
        if self.is_empty() || rhs.is_empty() {
            *self = Self::empty();
        } else {
            let mut result = HashMap::new();
            for (lhs, lhs_count) in &self.contents {
                for (rhs, rhs_count) in &rhs.contents {
                    *result.entry(lhs + rhs).or_insert(Count::zero()) +=
                        lhs_count * rhs_count;
                }
            }
            self.contents = result;
            self.count *= &rhs.count;
        }
    }
}
impl Mul<Distribution> for &Distribution {
    type Output = Distribution;

    fn mul(self, rhs: Distribution) -> Self::Output {
        rhs * self
    }
}
impl Mul<&Distribution> for &Distribution {
    type Output = Distribution;

    fn mul(self, rhs: &Distribution) -> Self::Output {
        let mut result = HashMap::new();
        if self.is_empty() || rhs.is_empty() {
            return Distribution::empty();
        }
        for (lhs, lhs_count) in &self.contents {
            for (rhs, rhs_count) in &rhs.contents {
                *result.entry(lhs + rhs).or_insert(Count::zero()) +=
                    lhs_count * rhs_count;
            }
        }
        Distribution::new(result, &self.count * &rhs.count)
    }
}

// OR is the only operation that can be done with empty distributions -
// for all other operations the result is empty if any of the operands is empty
impl BitOr<Distribution> for Distribution {
    type Output = Distribution;

    fn bitor(self, rhs: Distribution) -> Self::Output {
        self | &rhs
    }
}
impl BitOr<Distribution> for &Distribution {
    type Output = Distribution;

    fn bitor(self, rhs: Distribution) -> Self::Output {
        rhs | self
    }
}
impl BitOr<&Distribution> for Distribution {
    type Output = Distribution;

    fn bitor(mut self, rhs: &Distribution) -> Self::Output {
        self |= rhs;
        self
    }
}
impl BitOr<&Distribution> for &Distribution {
    type Output = Distribution;

    fn bitor(self, rhs: &Distribution) -> Self::Output {
        if self.is_empty() {
            return rhs.clone();
        } else if rhs.is_empty() {
            return self.clone();
        }
        let mut result = self.clone();
        for (entry, count) in &rhs.contents {
            *result
                .contents
                .entry(entry.clone())
                .or_insert(Count::zero()) += count;
        }
        result.count += &rhs.count;
        result
    }
}
impl BitOrAssign<Distribution> for Distribution {
    fn bitor_assign(&mut self, rhs: Distribution) {
        if self.is_empty() {
            *self = rhs;
        } else {
            *self |= &rhs;
        }
    }
}
impl BitOrAssign<&Distribution> for Distribution {
    fn bitor_assign(&mut self, rhs: &Distribution) {
        if rhs.is_empty() {
            return;
        } else if self.is_empty() {
            *self = rhs.clone();
            return;
        }
        for (result, count) in &rhs.contents {
            *self.contents.entry(result.clone()).or_insert(Count::zero()) += count;
        }
        self.count += &rhs.count;
    }
}
impl Neg for Distribution {
    type Output = Distribution;

    fn neg(mut self) -> Self::Output {
        self.contents = self.contents.into_iter().map(|(k, v)| (-k, v)).collect();
        self
    }
}

impl Distribution {
    pub fn new(contents: HashMap<Value, Count>, total_size: Count) -> Self {
        debug_assert_eq!(contents.values().sum::<Count>(), total_size);
        Self {
            contents,
            count: total_size,
        }
    }

    pub fn empty() -> Self {
        Self {
            contents: HashMap::new(),
            count: Count::zero(),
        }
    }

    pub fn into_raw(self) -> (Vec<(Value, Count)>, Count) {
        let mut contents = self.contents.into_iter().collect_vec();
        contents.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
        (contents, self.count)
    }

    pub fn into_approximate(self) -> Vec<(Value, f64)> {
        let total = self.count.to_f64().unwrap();
        let mut result = self
            .contents
            .into_iter()
            .map(|(k, v)| (k, v.to_f64().unwrap() / total))
            .collect_vec();
        result.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
        result
    }
}

#[derive(Debug, Clone)]
pub enum RollError {
    NonPositive(Value),
    FractionalSides(Value),
}

impl Distribution {
    fn roll_die(sides: &Value, scale: &Count) -> Result<Self, RollError> {
        if sides <= &Ratio::new(0.into(), 1.into()) {
            return Err(RollError::NonPositive(sides.clone()));
        } else if !sides.is_integer() {
            return Err(RollError::FractionalSides(sides.clone()));
        }
        let mut result = Self::new(
            range(1.into(), sides.to_integer() + 1)
                .map(|i| (Value::new(i, 1.into()), uint(1)))
                .collect(),
            sides.to_integer().to_biguint().unwrap(),
        );
        result.scale_count(scale);
        Ok(result)
    }

    pub fn uniform_over(range: &Self) -> Result<Self, RollError> {
        let mut result = Self::empty();
        if range.is_empty() {
            return Ok(result);
        }
        for (sides, count) in &range.contents {
            result |= Self::roll_die(sides, count)?;
        }
        result.simplify();
        Ok(result)
    }

    pub fn is_empty(&self) -> bool {
        self.count == Count::zero()
    }

    pub fn simplify(&mut self) {
        debug_assert_eq!(self.contents.values().sum::<Count>(), self.count);
        if self.is_empty() {
            return;
        }
        let mut gcd = self.count.clone();
        self.contents.retain(|_, v| {
            if v == &Count::zero() {
                false
            } else {
                gcd = gcd.gcd(v);
                true
            }
        });
        if gcd != uint(1) {
            self.contents.values_mut().for_each(|v| *v /= &gcd);
            self.count /= gcd;
        }
        debug_assert_eq!(self.contents.values().sum::<Count>(), self.count);
    }

    fn scale_count(&mut self, scale: &Count) {
        self.contents.values_mut().for_each(|v| *v *= scale);
        self.count *= scale;
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::enum_variant_names)]
pub enum RepeatError {
    FractionalRepeatCount(Value),
    NegativeRepeatCount(Value),
    FractionalKeepCount(Value),
    NegativeKeepCount(Value),
}

impl Distribution {
    fn repeat_once(
        &self,
        times: &Value,
        scale: &Count,
    ) -> Result<Distribution, RepeatError> {
        if !times.is_integer() {
            Err(RepeatError::FractionalRepeatCount(times.clone()))
        } else if times < &Value::zero() {
            Err(RepeatError::NegativeRepeatCount(times.clone()))
        } else if times == &Value::zero() {
            Ok(Self::empty())
        } else {
            let mut result = self.clone();
            for _ in range(1.into(), times.to_integer()) {
                result += self;
            }
            result.scale_count(scale);
            Ok(result)
        }
    }

    pub fn repeat(&self, times: &Distribution) -> Result<Distribution, RepeatError> {
        let mut result = Self::empty();
        if self.is_empty() || times.is_empty() {
            return Ok(result);
        }
        for (times_result, times_count) in &times.contents {
            result |= self.repeat_once(times_result, times_count)?;
        }
        result.simplify();
        Ok(result)
    }

    fn repeat_once_advantage(
        &self,
        times: &Value,
        keep_highest: &Value,
        scale: Count,
    ) -> Result<Distribution, RepeatError> {
        if !times.is_integer() {
            return Err(RepeatError::FractionalRepeatCount(times.clone()));
        }
        let Some(times) = times.to_integer().to_biguint() else {
            return Err(RepeatError::NegativeRepeatCount(times.clone()));
        };
        if !keep_highest.is_integer() {
            return Err(RepeatError::FractionalKeepCount(keep_highest.clone()));
        }
        let keep_highest =
            if let Some(keep_highest) = keep_highest.to_integer().to_biguint() {
                keep_highest
                    .try_into()
                    .expect("keep_highest is ridiculously large")
            } else {
                return Err(RepeatError::NegativeKeepCount(keep_highest.clone()));
            };

        let mut result = Self::empty();
        for mut rolls in range(Count::zero(), times.clone())
            .map(|_| &self.contents)
            .multi_cartesian_product()
        {
            let count = rolls.iter().map(|(_, v)| *v).product::<Count>();
            rolls.sort_unstable_by_key(|(k, _)| Reverse(*k));
            let value = rolls.into_iter().take(keep_highest).map(|(k, _)| k).sum();
            *result.contents.entry(value).or_insert(Count::zero()) += count;
        }
        result.count = Pow::pow(&self.count, times) * scale;
        Ok(result)
    }

    pub fn repeat_advantage(
        &self,
        times: &Distribution,
        keep_highest: &Distribution,
    ) -> Result<Distribution, RepeatError> {
        let mut result = Self::empty();
        if self.is_empty() || times.is_empty() || keep_highest.is_empty() {
            return Ok(result);
        }
        for (times_result, times_count) in &times.contents {
            for (keep_result, keep_count) in &keep_highest.contents {
                result |= self.repeat_once_advantage(
                    times_result,
                    keep_result,
                    times_count * keep_count,
                )?;
            }
        }
        result.simplify();
        Ok(result)
    }

    fn repeat_once_disadvantage(
        &self,
        times: &Value,
        keep_lowest: &Value,
        scale: Count,
    ) -> Result<Distribution, RepeatError> {
        if !times.is_integer() {
            return Err(RepeatError::FractionalRepeatCount(times.clone()));
        }
        let Some(times) = times.to_integer().to_biguint() else {
            return Err(RepeatError::NegativeRepeatCount(times.clone()));
        };
        if !keep_lowest.is_integer() {
            return Err(RepeatError::FractionalKeepCount(keep_lowest.clone()));
        }
        let keep_lowest =
            if let Some(keep_lowest) = keep_lowest.to_integer().to_biguint() {
                keep_lowest
                    .try_into()
                    .expect("keep_lowest is ridiculously large")
            } else {
                return Err(RepeatError::NegativeKeepCount(keep_lowest.clone()));
            };

        let mut result = Self::empty();
        for mut rolls in range(Count::zero(), times.clone())
            .map(|_| &self.contents)
            .multi_cartesian_product()
        {
            let count = rolls.iter().map(|(_, v)| *v).product::<Count>();
            rolls.sort_unstable_by_key(|(k, _)| *k);
            let value = rolls.into_iter().take(keep_lowest).map(|(k, _)| k).sum();
            *result.contents.entry(value).or_insert(Count::zero()) += count;
        }
        result.count = Pow::pow(&self.count, times) * scale;
        Ok(result)
    }

    pub fn repeat_disadvantage(
        &self,
        times: &Distribution,
        keep_lowest: &Distribution,
    ) -> Result<Distribution, RepeatError> {
        let mut result = Self::empty();
        if self.is_empty() || times.is_empty() || keep_lowest.is_empty() {
            return Ok(result);
        }
        for (times_result, times_count) in &times.contents {
            for (keep_result, keep_count) in &keep_lowest.contents {
                result |= self.repeat_once_disadvantage(
                    times_result,
                    keep_result,
                    times_count * keep_count,
                )?;
            }
        }
        result.simplify();
        Ok(result)
    }

    pub fn max(lhs: &Self, rhs: &Self) -> Self {
        if lhs.is_empty() || rhs.is_empty() {
            return Self::empty();
        }
        let mut result = HashMap::new();
        for (lhs, lhs_count) in &lhs.contents {
            for (rhs, rhs_count) in &rhs.contents {
                *result
                    .entry(Ord::max(lhs, rhs).clone())
                    .or_insert(Count::zero()) += lhs_count * rhs_count;
            }
        }
        let mut result = Self::new(result, &lhs.count * &rhs.count);
        result.simplify();
        result
    }

    pub fn min(lhs: &Self, rhs: &Self) -> Self {
        if lhs.is_empty() || rhs.is_empty() {
            return Self::empty();
        }
        let mut result = HashMap::new();
        for (lhs, lhs_count) in &lhs.contents {
            for (rhs, rhs_count) in &rhs.contents {
                *result
                    .entry(Ord::min(lhs, rhs).clone())
                    .or_insert(Count::zero()) += lhs_count * rhs_count;
            }
        }
        let mut result = Self::new(result, &lhs.count * &rhs.count);
        result.simplify();
        result
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DivideByZeroError;

impl Distribution {
    pub fn true_div(&self, divisor: &Self) -> Result<Self, DivideByZeroError> {
        if self.is_empty() || divisor.is_empty() {
            return Ok(Self::empty());
        }
        let mut result = HashMap::new();
        for (lhs, lhs_count) in &self.contents {
            for (rhs, rhs_count) in &divisor.contents {
                if rhs == &Value::zero() {
                    return Err(DivideByZeroError);
                }
                *result.entry(lhs / rhs).or_insert(Count::zero()) +=
                    lhs_count * rhs_count;
            }
        }
        let mut result = Self::new(result, &self.count * &divisor.count);
        result.simplify();
        Ok(result)
    }

    pub fn trunc_div(&self, divisor: &Self) -> Result<Self, DivideByZeroError> {
        if self.is_empty() || divisor.is_empty() {
            return Ok(Self::empty());
        }
        let mut result = HashMap::new();
        for (lhs, lhs_count) in &self.contents {
            for (rhs, rhs_count) in &divisor.contents {
                if rhs == &Value::zero() {
                    return Err(DivideByZeroError);
                }
                *result.entry((lhs / rhs).trunc()).or_insert(Count::zero()) +=
                    lhs_count * rhs_count;
            }
        }
        let mut result = Self::new(result, &self.count * &divisor.count);
        result.simplify();
        Ok(result)
    }

    pub fn floor_div(&self, divisor: &Self) -> Result<Self, DivideByZeroError> {
        if self.is_empty() || divisor.is_empty() {
            return Ok(Self::empty());
        }
        let mut result = HashMap::new();
        for (lhs, lhs_count) in &self.contents {
            for (rhs, rhs_count) in &divisor.contents {
                if rhs == &Value::zero() {
                    return Err(DivideByZeroError);
                }
                *result.entry((lhs / rhs).floor()).or_insert(Count::zero()) +=
                    lhs_count * rhs_count;
            }
        }
        let mut result = Self::new(result, &self.count * &divisor.count);
        result.simplify();
        Ok(result)
    }

    pub fn ceil_div(&self, divisor: &Self) -> Result<Self, DivideByZeroError> {
        if self.is_empty() || divisor.is_empty() {
            return Ok(Self::empty());
        }
        let mut result = HashMap::new();
        for (lhs, lhs_count) in &self.contents {
            for (rhs, rhs_count) in &divisor.contents {
                if rhs == &Value::zero() {
                    return Err(DivideByZeroError);
                }
                *result.entry((lhs / rhs).ceil()).or_insert(Count::zero()) +=
                    lhs_count * rhs_count;
            }
        }
        let mut result = Self::new(result, &self.count * &divisor.count);
        result.simplify();
        Ok(result)
    }

    pub fn modulo(&self, divisor: &Self) -> Result<Self, DivideByZeroError> {
        if self.is_empty() || divisor.is_empty() {
            return Ok(Self::empty());
        }
        let mut result = HashMap::new();
        for (lhs, lhs_count) in &self.contents {
            for (rhs, rhs_count) in &divisor.contents {
                if rhs == &Value::zero() {
                    return Err(DivideByZeroError);
                }
                *result
                    .entry((lhs % rhs + rhs) % rhs)
                    .or_insert(Count::zero()) += lhs_count * rhs_count;
            }
        }
        let mut result = Self::new(result, &self.count * &divisor.count);
        result.simplify();
        Ok(result)
    }
}

#[derive(Debug, Clone)]
pub enum PowError {
    DivideByZero,
    FractionalPower(Value),
    PowerError(TryFromIntError),
}
impl From<TryFromIntError> for PowError {
    fn from(error: TryFromIntError) -> Self {
        PowError::PowerError(error)
    }
}

impl Distribution {
    pub fn pow(&self, exponent: &Self) -> Result<Self, PowError> {
        if self.is_empty() || exponent.is_empty() {
            return Ok(Self::empty());
        }
        let mut new_distribution = HashMap::new();
        for (lhs, count) in &self.contents {
            for (rhs, exponent_count) in &exponent.contents {
                if lhs == &Value::zero() && rhs < &Value::zero() {
                    return Err(PowError::DivideByZero);
                } else if !rhs.is_integer() {
                    return Err(PowError::FractionalPower(rhs.clone()));
                }
                let rhs = rhs.to_integer();
                let result = if rhs.is_negative() {
                    Value::new(
                        Pow::pow(lhs.denom(), (-&rhs).to_biguint().unwrap()),
                        Pow::pow(lhs.numer(), (-rhs).to_biguint().unwrap()),
                    )
                } else {
                    Value::new(
                        Pow::pow(lhs.numer(), rhs.to_biguint().unwrap()),
                        Pow::pow(lhs.denom(), rhs.to_biguint().unwrap()),
                    )
                };
                *new_distribution.entry(result).or_insert(Count::zero()) +=
                    count * exponent_count;
            }
        }
        let mut result = Self::new(new_distribution, &self.count * &exponent.count);
        result.simplify();
        Ok(result)
    }
}

#[derive(Debug, Clone)]
pub enum SqrtError {
    IrrationalResult(Value),
    NegativeArgument(Value),
}

impl Distribution {
    pub fn sqrt(&self) -> Result<Self, SqrtError> {
        if self.is_empty() {
            return Ok(Self::empty());
        }
        let mut result = HashMap::new();
        for (arg, count) in &self.contents {
            if arg.is_negative() {
                return Err(SqrtError::NegativeArgument(arg.clone()));
            }
            let sqrt = Value::new(arg.numer().sqrt(), arg.denom().sqrt());
            if &(&sqrt * &sqrt) != arg {
                return Err(SqrtError::IrrationalResult(arg.clone()));
            }
            *result.entry(sqrt).or_insert(Count::zero()) += count;
        }
        let mut result = Self::new(result, self.count.clone());
        result.simplify();
        Ok(result)
    }

    pub fn floor(&self) -> Self {
        if self.is_empty() {
            return Self::empty();
        }
        let mut result = HashMap::new();
        for (arg, count) in &self.contents {
            *result.entry(arg.floor()).or_insert(Count::zero()) += count;
        }
        let mut result = Self::new(result, self.count.clone());
        result.simplify();
        result
    }

    pub fn ceil(&self) -> Self {
        if self.is_empty() {
            return Self::empty();
        }
        let mut result = HashMap::new();
        for (arg, count) in &self.contents {
            *result.entry(arg.ceil()).or_insert(Count::zero()) += count;
        }
        let mut result = Self::new(result, self.count.clone());
        result.simplify();
        result
    }

    pub fn round(&self) -> Self {
        if self.is_empty() {
            return Self::empty();
        }
        let mut result = HashMap::new();
        for (arg, count) in &self.contents {
            *result.entry(arg.round()).or_insert(Count::zero()) += count;
        }
        let mut result = Self::new(result, self.count.clone());
        result.simplify();
        result
    }

    pub fn trunc(&self) -> Self {
        if self.is_empty() {
            return Self::empty();
        }
        let mut result = HashMap::new();
        for (arg, count) in &self.contents {
            *result.entry(arg.trunc()).or_insert(Count::zero()) += count;
        }
        let mut result = Self::new(result, self.count.clone());
        result.simplify();
        result
    }

    pub fn abs(&self) -> Self {
        if self.is_empty() {
            return Self::empty();
        }
        let mut result = HashMap::new();
        for (arg, count) in &self.contents {
            *result.entry(arg.abs()).or_insert(Count::zero()) += count;
        }
        let mut result = Self::new(result, self.count.clone());
        result.simplify();
        result
    }
}

impl Distribution {
    pub fn less_than(&self, rhs: &Self) -> Self {
        if self.is_empty() || rhs.is_empty() {
            return Self::empty();
        }
        let mut true_values = Count::zero();
        let mut false_values = Count::zero();
        for (lhs, lhs_count) in &self.contents {
            for (rhs, rhs_count) in &rhs.contents {
                if lhs < rhs {
                    true_values += lhs_count * rhs_count;
                } else {
                    false_values += lhs_count * rhs_count;
                }
            }
        }
        if true_values.is_zero() {
            Self::from(0)
        } else if false_values.is_zero() {
            Self::from(1)
        } else {
            let gcd = true_values.gcd(&false_values);
            let true_values = true_values / &gcd;
            let false_values = false_values / gcd;
            let total = &true_values + &false_values;
            Self::new(
                [
                    (Value::zero(), false_values),
                    (Value::new(1.into(), 1.into()), true_values),
                ]
                .into(),
                total,
            )
        }
    }

    pub fn less_equal(&self, rhs: &Self) -> Self {
        if self.is_empty() || rhs.is_empty() {
            return Self::empty();
        }
        let mut true_values = Count::zero();
        let mut false_values = Count::zero();
        for (lhs, lhs_count) in &self.contents {
            for (rhs, rhs_count) in &rhs.contents {
                if lhs <= rhs {
                    true_values += lhs_count * rhs_count;
                } else {
                    false_values += lhs_count * rhs_count;
                }
            }
        }
        if true_values.is_zero() {
            Self::from(0)
        } else if false_values.is_zero() {
            Self::from(1)
        } else {
            let gcd = true_values.gcd(&false_values);
            let true_values = true_values / &gcd;
            let false_values = false_values / gcd;
            let total = &true_values + &false_values;
            Self::new(
                [
                    (Value::zero(), false_values),
                    (Value::new(1.into(), 1.into()), true_values),
                ]
                .into(),
                total,
            )
        }
    }

    pub fn greater_than(&self, rhs: &Self) -> Self {
        if self.is_empty() || rhs.is_empty() {
            return Self::empty();
        }
        let mut true_values = Count::zero();
        let mut false_values = Count::zero();
        for (lhs, lhs_count) in &self.contents {
            for (rhs, rhs_count) in &rhs.contents {
                if lhs > rhs {
                    true_values += lhs_count * rhs_count;
                } else {
                    false_values += lhs_count * rhs_count;
                }
            }
        }
        if true_values.is_zero() {
            Self::from(0)
        } else if false_values.is_zero() {
            Self::from(1)
        } else {
            let gcd = true_values.gcd(&false_values);
            let true_values = true_values / &gcd;
            let false_values = false_values / gcd;
            let total = &true_values + &false_values;
            Self::new(
                [
                    (Value::zero(), false_values),
                    (Value::new(1.into(), 1.into()), true_values),
                ]
                .into(),
                total,
            )
        }
    }

    pub fn greater_equal(&self, rhs: &Self) -> Self {
        if self.is_empty() || rhs.is_empty() {
            return Self::empty();
        }
        let mut true_values = Count::zero();
        let mut false_values = Count::zero();
        for (lhs, lhs_count) in &self.contents {
            for (rhs, rhs_count) in &rhs.contents {
                if lhs >= rhs {
                    true_values += lhs_count * rhs_count;
                } else {
                    false_values += lhs_count * rhs_count;
                }
            }
        }
        if true_values.is_zero() {
            Self::from(0)
        } else if false_values.is_zero() {
            Self::from(1)
        } else {
            let gcd = true_values.gcd(&false_values);
            let true_values = true_values / &gcd;
            let false_values = false_values / gcd;
            let total = &true_values + &false_values;
            Self::new(
                [
                    (Value::zero(), false_values),
                    (Value::new(1.into(), 1.into()), true_values),
                ]
                .into(),
                total,
            )
        }
    }

    pub fn equal(&self, rhs: &Self) -> Self {
        if self.is_empty() || rhs.is_empty() {
            return Self::empty();
        }
        let mut true_values = Count::zero();
        let mut false_values = Count::zero();
        for (lhs, lhs_count) in &self.contents {
            for (rhs, rhs_count) in &rhs.contents {
                if lhs == rhs {
                    true_values += lhs_count * rhs_count;
                } else {
                    false_values += lhs_count * rhs_count;
                }
            }
        }
        if true_values.is_zero() {
            Self::from(0)
        } else if false_values.is_zero() {
            Self::from(1)
        } else {
            let gcd = true_values.gcd(&false_values);
            let true_values = true_values / &gcd;
            let false_values = false_values / gcd;
            let total = &true_values + &false_values;
            Self::new(
                [
                    (Value::zero(), false_values),
                    (Value::new(1.into(), 1.into()), true_values),
                ]
                .into(),
                total,
            )
        }
    }
}
