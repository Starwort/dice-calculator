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
use num::rational::Ratio;
use num::{Integer, Signed};

pub type Value = Ratio<isize>;

#[derive(Debug, Clone)]
pub struct Distribution {
    pub contents: HashMap<Value, usize>,
    pub count: usize,
}
impl From<isize> for Distribution {
    fn from(value: isize) -> Self {
        Self::new([(Ratio::new(value, 1), 1)].into(), 1)
    }
}
impl From<Value> for Distribution {
    fn from(value: Value) -> Self {
        Self::new([(value, 1)].into(), 1)
    }
}
impl Sum for Distribution {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::new(HashMap::new(), 0), Add::add)
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
                *result.entry(lhs + rhs).or_insert(0) += lhs_count * rhs_count;
            }
        }
        Distribution::new(result, self.count * rhs.count)
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
                *result.entry(lhs + rhs).or_insert(0) += lhs_count * rhs_count;
            }
        }
        self.contents = result;
        self.count *= rhs.count;
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
        let mut resul = HashMap::new();
        for (lhs, lhs_count) in &self.contents {
            for (rhs, rhs_count) in &rhs.contents {
                *resul.entry(lhs - rhs).or_insert(0) += lhs_count * rhs_count;
            }
        }
        self.contents = resul;
        self.count *= rhs.count;
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
                *result.entry(lhs - rhs).or_insert(0) += lhs_count * rhs_count;
            }
        }
        Distribution::new(result, self.count * rhs.count)
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
                    *result.entry(lhs + rhs).or_insert(0) += lhs_count * rhs_count;
                }
            }
            self.contents = result;
            self.count *= rhs.count;
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
                *result.entry(lhs + rhs).or_insert(0) += lhs_count * rhs_count;
            }
        }
        Distribution::new(result, self.count * rhs.count)
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
        for (&entry, count) in &rhs.contents {
            *result.contents.entry(entry).or_insert(0) += count;
        }
        result.count += rhs.count;
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
            *self.contents.entry(*result).or_insert(0) += count;
        }
        self.count += rhs.count;
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
    pub fn new(contents: HashMap<Value, usize>, total_size: usize) -> Self {
        debug_assert_eq!(contents.values().sum::<usize>(), total_size);
        Self {
            contents,
            count: total_size,
        }
    }

    pub fn empty() -> Self {
        Self {
            contents: HashMap::new(),
            count: 0,
        }
    }

    pub fn into_raw(self) -> (Vec<(Value, usize)>, usize) {
        let mut contents = self.contents.into_iter().collect_vec();
        contents.sort_unstable_by_key(|(k, _)| *k);
        (contents, self.count)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum RollError {
    NonPositive(Value),
    FractionalSides(Value),
}

impl Distribution {
    fn roll_die(sides: Value, scale: usize) -> Result<Self, RollError> {
        if sides <= Ratio::new(0, 1) {
            return Err(RollError::NonPositive(sides));
        } else if !sides.is_integer() {
            return Err(RollError::FractionalSides(sides));
        }
        let mut result = Self::new(
            (1..=(sides.to_integer()))
                .map(|i| (Ratio::new(i, 1), 1))
                .collect(),
            #[allow(clippy::cast_sign_loss)]
            {
                sides.to_integer() as usize
            },
        );
        result.scale_count(scale);
        Ok(result)
    }

    pub fn uniform_over(range: &Self) -> Result<Self, RollError> {
        let mut result = Self::empty();
        if range.is_empty() {
            return Ok(result);
        }
        for (&sides, &count) in &range.contents {
            result |= Self::roll_die(sides, count)?;
        }
        result.simplify();
        Ok(result)
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn simplify(&mut self) {
        if self.is_empty() {
            return;
        }
        let mut gcd = self.count;
        self.contents.retain(|_, &mut v| {
            if v == 0 {
                false
            } else {
                gcd = gcd.gcd(&v);
                true
            }
        });
        if gcd != 1 {
            self.contents.values_mut().for_each(|v| *v /= gcd);
            self.count /= gcd;
        }
    }

    fn scale_count(&mut self, scale: usize) {
        self.contents.values_mut().for_each(|v| *v *= scale);
        self.count *= scale;
    }
}

#[derive(Debug, Clone, Copy)]
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
        times: Value,
        scale: usize,
    ) -> Result<Distribution, RepeatError> {
        if !times.is_integer() {
            Err(RepeatError::FractionalRepeatCount(times))
        } else if times < Ratio::new(0, 1) {
            Err(RepeatError::NegativeRepeatCount(times))
        } else if times == Ratio::new(0, 1) {
            Ok(Self::empty())
        } else {
            let mut result = self.clone();
            for _ in 1..times.to_integer() {
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
        for (&times_result, &times_count) in &times.contents {
            result |= self.repeat_once(times_result, times_count)?;
        }
        result.simplify();
        Ok(result)
    }

    fn repeat_once_advantage(
        &self,
        times: Value,
        keep_highest: Value,
        scale: usize,
    ) -> Result<Distribution, RepeatError> {
        if !times.is_integer() {
            return Err(RepeatError::FractionalRepeatCount(times));
        } else if times < Ratio::new(0, 1) {
            return Err(RepeatError::NegativeRepeatCount(times));
        }
        if !keep_highest.is_integer() {
            return Err(RepeatError::FractionalKeepCount(keep_highest));
        } else if keep_highest < Ratio::new(0, 1) {
            return Err(RepeatError::NegativeKeepCount(keep_highest));
        }
        #[allow(clippy::cast_sign_loss)]
        let keep_highest = keep_highest.to_integer() as usize;
        let mut result = Self::empty();
        for mut rolls in (0..times.to_integer())
            .map(|_| &self.contents)
            .multi_cartesian_product()
        {
            let count = rolls.iter().map(|(_, &v)| v).product::<usize>();
            rolls.sort_unstable_by_key(|(k, _)| Reverse(*k));
            let value = rolls.into_iter().take(keep_highest).map(|(k, _)| k).sum();
            *result.contents.entry(value).or_insert(0) += count;
        }
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        {
            result.count = self.count.pow(times.to_integer() as u32) * scale;
        }
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
        for (&times_result, &times_count) in &times.contents {
            for (&keep_result, &keep_count) in &keep_highest.contents {
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
        times: Value,
        keep_lowest: Value,
        scale: usize,
    ) -> Result<Distribution, RepeatError> {
        if !times.is_integer() {
            return Err(RepeatError::FractionalRepeatCount(times));
        } else if times < Ratio::new(0, 1) {
            return Err(RepeatError::NegativeRepeatCount(times));
        }
        if !keep_lowest.is_integer() {
            return Err(RepeatError::FractionalKeepCount(keep_lowest));
        } else if keep_lowest < Ratio::new(0, 1) {
            return Err(RepeatError::NegativeKeepCount(keep_lowest));
        }
        #[allow(clippy::cast_sign_loss)]
        let keep_lowest = keep_lowest.to_integer() as usize;
        let mut result = Self::empty();
        for mut rolls in (0..times.to_integer())
            .map(|_| &self.contents)
            .multi_cartesian_product()
        {
            let count = rolls.iter().map(|(_, &v)| v).product::<usize>();
            rolls.sort_unstable_by_key(|(k, _)| *k);
            let value = rolls.into_iter().take(keep_lowest).map(|(k, _)| k).sum();
            *result.contents.entry(value).or_insert(0) += count;
        }
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        {
            result.count = self.count.pow(times.to_integer() as u32) * scale;
        }
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
        for (&times_result, &times_count) in &times.contents {
            for (&keep_result, &keep_count) in &keep_lowest.contents {
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

    pub fn max(a: &Self, b: &Self) -> Self {
        if a.is_empty() || b.is_empty() {
            return Self::empty();
        }
        let mut new_distribution = HashMap::new();
        for (&result, &count) in &a.contents {
            for (&b_result, &b_count) in &b.contents {
                *new_distribution
                    .entry(Value::max(result, b_result))
                    .or_insert(0) += count * b_count;
            }
        }
        let mut result = Self::new(new_distribution, a.count * b.count);
        result.simplify();
        result
    }

    pub fn min(a: &Self, b: &Self) -> Self {
        if a.is_empty() || b.is_empty() {
            return Self::empty();
        }
        let mut new_distribution = HashMap::new();
        for (&result, &count) in &a.contents {
            for (&b_result, &b_count) in &b.contents {
                *new_distribution
                    .entry(Value::min(result, b_result))
                    .or_insert(0) += count * b_count;
            }
        }
        let mut result = Self::new(new_distribution, a.count * b.count);
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
        let mut new_distribution = HashMap::new();
        for (&result, &count) in &self.contents {
            for (&divisor_result, &divisor_count) in &divisor.contents {
                if divisor_result == Ratio::new(0, 1) {
                    return Err(DivideByZeroError);
                }
                *new_distribution.entry(result / divisor_result).or_insert(0) +=
                    count * divisor_count;
            }
        }
        let mut result = Self::new(new_distribution, self.count * divisor.count);
        result.simplify();
        Ok(result)
    }

    pub fn trunc_div(&self, divisor: &Self) -> Result<Self, DivideByZeroError> {
        if self.is_empty() || divisor.is_empty() {
            return Ok(Self::empty());
        }
        let mut new_distribution = HashMap::new();
        for (&result, &count) in &self.contents {
            for (&divisor_result, &divisor_count) in &divisor.contents {
                if divisor_result == Ratio::new(0, 1) {
                    return Err(DivideByZeroError);
                }
                *new_distribution
                    .entry((result / divisor_result).trunc())
                    .or_insert(0) += count * divisor_count;
            }
        }
        let mut result = Self::new(new_distribution, self.count * divisor.count);
        result.simplify();
        Ok(result)
    }

    pub fn floor_div(&self, divisor: &Self) -> Result<Self, DivideByZeroError> {
        if self.is_empty() || divisor.is_empty() {
            return Ok(Self::empty());
        }
        let mut new_distribution = HashMap::new();
        for (&result, &count) in &self.contents {
            for (&divisor_result, &divisor_count) in &divisor.contents {
                if divisor_result == Ratio::new(0, 1) {
                    return Err(DivideByZeroError);
                }
                *new_distribution
                    .entry((result / divisor_result).floor())
                    .or_insert(0) += count * divisor_count;
            }
        }
        let mut result = Self::new(new_distribution, self.count * divisor.count);
        result.simplify();
        Ok(result)
    }

    pub fn ceil_div(&self, divisor: &Self) -> Result<Self, DivideByZeroError> {
        if self.is_empty() || divisor.is_empty() {
            return Ok(Self::empty());
        }
        let mut new_distribution = HashMap::new();
        for (&result, &count) in &self.contents {
            for (&divisor_result, &divisor_count) in &divisor.contents {
                if divisor_result == Ratio::new(0, 1) {
                    return Err(DivideByZeroError);
                }
                *new_distribution
                    .entry((result / divisor_result).ceil())
                    .or_insert(0) += count * divisor_count;
            }
        }
        let mut result = Self::new(new_distribution, self.count * divisor.count);
        result.simplify();
        Ok(result)
    }

    pub fn modulo(&self, divisor: &Self) -> Result<Self, DivideByZeroError> {
        if self.is_empty() || divisor.is_empty() {
            return Ok(Self::empty());
        }
        let mut new_distribution = HashMap::new();
        for (&result, &count) in &self.contents {
            for (&divisor_result, &divisor_count) in &divisor.contents {
                if divisor_result == Ratio::new(0, 1) {
                    return Err(DivideByZeroError);
                }
                let new_result = result % divisor_result;
                *new_distribution
                    .entry((new_result + divisor_result) % divisor_result)
                    .or_insert(0) += count * divisor_count;
            }
        }
        let mut result = Self::new(new_distribution, self.count * divisor.count);
        result.simplify();
        Ok(result)
    }
}

#[derive(Debug, Clone, Copy)]
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
        for (&result, &count) in &self.contents {
            for (&exponent_result, &exponent_count) in &exponent.contents {
                if result == Ratio::new(0, 1) && exponent_result < Ratio::new(0, 1) {
                    return Err(PowError::DivideByZero);
                } else if !exponent_result.is_integer() {
                    return Err(PowError::FractionalPower(exponent_result));
                }
                *new_distribution
                    .entry(result.pow(i32::try_from(exponent_result.to_integer())?))
                    .or_insert(0) += count * exponent_count;
            }
        }
        let mut result = Self::new(new_distribution, self.count * exponent.count);
        result.simplify();
        Ok(result)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SqrtError {
    IrrationalResult(Value),
    NegativeArgument(Value),
}

impl Distribution {
    pub fn sqrt(&self) -> Result<Self, SqrtError> {
        if self.is_empty() {
            return Ok(Self::empty());
        }
        let mut new_distribution = HashMap::new();
        for (&result, &count) in &self.contents {
            if result < Ratio::new(0, 1) {
                return Err(SqrtError::NegativeArgument(result));
            }
            let sqrt = Value::new(
                result
                    .numer()
                    .checked_isqrt()
                    .ok_or(SqrtError::IrrationalResult(result))?,
                result
                    .denom()
                    .checked_isqrt()
                    .ok_or(SqrtError::IrrationalResult(result))?,
            );
            *new_distribution.entry(sqrt).or_insert(0) += count;
        }
        let mut result = Self::new(new_distribution, self.count);
        result.simplify();
        Ok(result)
    }

    pub fn floor(&self) -> Self {
        if self.is_empty() {
            return Self::empty();
        }
        let mut new_distribution = HashMap::new();
        for (&result, &count) in &self.contents {
            *new_distribution.entry(result.floor()).or_insert(0) += count;
        }
        let mut result = Self::new(new_distribution, self.count);
        result.simplify();
        result
    }

    pub fn ceil(&self) -> Self {
        if self.is_empty() {
            return Self::empty();
        }
        let mut new_distribution = HashMap::new();
        for (&result, &count) in &self.contents {
            *new_distribution.entry(result.ceil()).or_insert(0) += count;
        }
        let mut result = Self::new(new_distribution, self.count);
        result.simplify();
        result
    }

    pub fn round(&self) -> Self {
        if self.is_empty() {
            return Self::empty();
        }
        let mut new_distribution = HashMap::new();
        for (&result, &count) in &self.contents {
            *new_distribution.entry(result.round()).or_insert(0) += count;
        }
        let mut result = Self::new(new_distribution, self.count);
        result.simplify();
        result
    }

    pub fn trunc(&self) -> Self {
        if self.is_empty() {
            return Self::empty();
        }
        let mut new_distribution = HashMap::new();
        for (&result, &count) in &self.contents {
            *new_distribution.entry(result.trunc()).or_insert(0) += count;
        }
        let mut result = Self::new(new_distribution, self.count);
        result.simplify();
        result
    }

    pub fn abs(&self) -> Self {
        if self.is_empty() {
            return Self::empty();
        }
        let mut new_distribution = HashMap::new();
        for (&result, &count) in &self.contents {
            *new_distribution.entry(result.abs()).or_insert(0) += count;
        }
        let mut result = Self::new(new_distribution, self.count);
        result.simplify();
        result
    }
}

impl Distribution {
    pub fn less_than(&self, rhs: &Self) -> Self {
        if self.is_empty() || rhs.is_empty() {
            return Self::empty();
        }
        let mut true_values = 0;
        let mut false_values = 0;
        for (&result, &count) in &self.contents {
            for (&rhs_result, &rhs_count) in &rhs.contents {
                if result < rhs_result {
                    true_values += count * rhs_count;
                } else {
                    false_values += count * rhs_count;
                }
            }
        }
        if true_values == 0 {
            Self::from(0)
        } else if false_values == 0 {
            Self::from(1)
        } else {
            let gcd = true_values.gcd(&false_values);
            let true_values = true_values / gcd;
            let false_values = false_values / gcd;
            Self::new(
                [
                    (Ratio::new(0, 1), false_values),
                    (Ratio::new(1, 1), true_values),
                ]
                .into(),
                true_values + false_values,
            )
        }
    }

    pub fn less_equal(&self, rhs: &Self) -> Self {
        if self.is_empty() || rhs.is_empty() {
            return Self::empty();
        }
        let mut true_values = 0;
        let mut false_values = 0;
        for (&result, &count) in &self.contents {
            for (&rhs_result, &rhs_count) in &rhs.contents {
                if result <= rhs_result {
                    true_values += count * rhs_count;
                } else {
                    false_values += count * rhs_count;
                }
            }
        }
        if true_values == 0 {
            Self::from(0)
        } else if false_values == 0 {
            Self::from(1)
        } else {
            let gcd = true_values.gcd(&false_values);
            let true_values = true_values / gcd;
            let false_values = false_values / gcd;
            Self::new(
                [
                    (Ratio::new(0, 1), false_values),
                    (Ratio::new(1, 1), true_values),
                ]
                .into(),
                true_values + false_values,
            )
        }
    }

    pub fn greater_than(&self, rhs: &Self) -> Self {
        if self.is_empty() || rhs.is_empty() {
            return Self::empty();
        }
        let mut true_values = 0;
        let mut false_values = 0;
        for (&result, &count) in &self.contents {
            for (&rhs_result, &rhs_count) in &rhs.contents {
                if result > rhs_result {
                    true_values += count * rhs_count;
                } else {
                    false_values += count * rhs_count;
                }
            }
        }
        if true_values == 0 {
            Self::from(0)
        } else if false_values == 0 {
            Self::from(1)
        } else {
            let gcd = true_values.gcd(&false_values);
            let true_values = true_values / gcd;
            let false_values = false_values / gcd;
            Self::new(
                [
                    (Ratio::new(0, 1), false_values),
                    (Ratio::new(1, 1), true_values),
                ]
                .into(),
                true_values + false_values,
            )
        }
    }

    pub fn greater_equal(&self, rhs: &Self) -> Self {
        if self.is_empty() || rhs.is_empty() {
            return Self::empty();
        }
        let mut true_values = 0;
        let mut false_values = 0;
        for (&result, &count) in &self.contents {
            for (&rhs_result, &rhs_count) in &rhs.contents {
                if result >= rhs_result {
                    true_values += count * rhs_count;
                } else {
                    false_values += count * rhs_count;
                }
            }
        }
        if true_values == 0 {
            Self::from(0)
        } else if false_values == 0 {
            Self::from(1)
        } else {
            let gcd = true_values.gcd(&false_values);
            let true_values = true_values / gcd;
            let false_values = false_values / gcd;
            Self::new(
                [
                    (Ratio::new(0, 1), false_values),
                    (Ratio::new(1, 1), true_values),
                ]
                .into(),
                true_values + false_values,
            )
        }
    }

    pub fn equal(&self, rhs: &Self) -> Self {
        if self.is_empty() || rhs.is_empty() {
            return Self::empty();
        }
        let mut true_values = 0;
        let mut false_values = 0;
        for (&result, &count) in &self.contents {
            for (&rhs_result, &rhs_count) in &rhs.contents {
                if result == rhs_result {
                    true_values += count * rhs_count;
                } else {
                    false_values += count * rhs_count;
                }
            }
        }
        if true_values == 0 {
            Self::from(0)
        } else if false_values == 0 {
            Self::from(1)
        } else {
            let gcd = true_values.gcd(&false_values);
            let true_values = true_values / gcd;
            let false_values = false_values / gcd;
            Self::new(
                [
                    (Ratio::new(0, 1), false_values),
                    (Ratio::new(1, 1), true_values),
                ]
                .into(),
                true_values + false_values,
            )
        }
    }
}
