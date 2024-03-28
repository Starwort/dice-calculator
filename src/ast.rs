use std::iter::once;

use from_pest::{ConversionError, FromPest, Void};
use itertools::Itertools;
use pest::iterators::Pairs;
use pest::Parser;
use pest_ast::FromPest;
use pest_derive::Parser;

use super::distribution::{Distribution, RepeatError};
use crate::distribution::{DivideByZeroError, PowError, RollError, SqrtError, Value};

#[derive(Debug, Clone, Copy)]
pub enum Error {
    RepeatError(RepeatError),
    DivideByZero,
    RollError(RollError),
    SqrtError(SqrtError),
    FunctionCallError(FunctionCallError),
    PowError(PowError),
}
impl From<RepeatError> for Error {
    fn from(e: RepeatError) -> Self {
        Error::RepeatError(e)
    }
}
impl From<DivideByZeroError> for Error {
    fn from(_: DivideByZeroError) -> Self {
        Error::DivideByZero
    }
}
impl From<RollError> for Error {
    fn from(e: RollError) -> Self {
        Error::RollError(e)
    }
}
impl From<SqrtError> for Error {
    fn from(e: SqrtError) -> Self {
        Error::SqrtError(e)
    }
}
impl From<FunctionCallError> for Error {
    fn from(e: FunctionCallError) -> Self {
        Error::FunctionCallError(e)
    }
}
impl From<PowError> for Error {
    fn from(e: PowError) -> Self {
        Error::PowError(e)
    }
}

#[derive(Parser)]
#[grammar = "dice.pest"]
struct DiceParser;

/// Parse the input string into an expression.
///
/// # Errors
///
/// If the input string is malformed.
pub fn parse(input: &str) -> Result<Expression, Box<pest::error::Error<Rule>>> {
    Ok(DiceParser::parse(Rule::Expression, input).map(|mut pairs| {
        Expression::from_pest(&mut pairs).unwrap_or_else(|err| unreachable!("{err}"))
    })?)
}

#[derive(FromPest, Debug)]
#[pest_ast(rule(Rule::Expression))]
pub struct Expression {
    expr: Expr,
}
impl Expression {
    /// Evaluate the expression and return the resulting distribution.
    ///
    /// # Errors
    ///
    /// Returns an error if the expression is unevaluable.
    pub fn eval(&self) -> Result<Distribution, Error> {
        self.expr.eval()
    }
}

#[derive(FromPest, Debug)]
#[pest_ast(rule(Rule::Expr))]
struct Expr {
    main_term: MinMaxTerm,
    comparisons: Vec<(CompOp, MinMaxTerm)>,
}
impl Expr {
    fn eval(&self) -> Result<Distribution, Error> {
        let main_term = self.main_term.eval()?;
        if self.comparisons.is_empty() {
            Ok(main_term)
        } else {
            let mut result = Distribution::from(1);
            let (operations_a, operations_b) = self
                .comparisons
                .iter()
                .map(|(op, term)| Ok::<_, Error>((op, term.eval()?)))
                .tee();
            for (lhs, maybe_rhs) in once(self.main_term.eval())
                .chain(operations_a.map(|term| Ok(term?.1)))
                .zip(operations_b)
            {
                let lhs = lhs?;
                let (op, rhs) = maybe_rhs?;
                result *= match op {
                    CompOp::LessThan => lhs.less_than(&rhs),
                    CompOp::LessThanEqual => lhs.less_equal(&rhs),
                    CompOp::GreaterThan => lhs.greater_than(&rhs),
                    CompOp::GreaterThanEqual => lhs.greater_equal(&rhs),
                    CompOp::Equal => lhs.equal(&rhs),
                };
            }
            Ok(result)
        }
    }
}

#[derive(Debug)]
enum CompOp {
    LessThan,
    LessThanEqual,
    GreaterThan,
    GreaterThanEqual,
    Equal,
}
impl<'pest> FromPest<'pest> for CompOp {
    type FatalError = Void;
    type Rule = Rule;

    fn from_pest(
        pest: &mut Pairs<'pest, Self::Rule>,
    ) -> Result<Self, ConversionError<Self::FatalError>> {
        match pest.peek() {
            Some(pair) => {
                match pair.as_rule() {
                    Rule::CompOp => {
                        match pest.next().unwrap().as_str() {
                            "<" => Ok(CompOp::LessThan),
                            "<=" => Ok(CompOp::LessThanEqual),
                            ">" => Ok(CompOp::GreaterThan),
                            ">=" => Ok(CompOp::GreaterThanEqual),
                            "=" => Ok(CompOp::Equal),
                            _ => unreachable!("No other strings matched by rule"),
                        }
                    },
                    _ => Err(ConversionError::NoMatch),
                }
            },
            None => Err(ConversionError::NoMatch),
        }
    }
}

#[derive(FromPest, Debug)]
#[pest_ast(rule(Rule::MinMaxTerm))]
struct MinMaxTerm {
    main_term: AddTerm,
    additions: Vec<(MinMaxOp, AddTerm)>,
}
impl MinMaxTerm {
    fn eval(&self) -> Result<Distribution, Error> {
        let mut result = self.main_term.eval()?;
        for (op, term) in &self.additions {
            result = match op {
                MinMaxOp::Max => Distribution::max(&result, &term.eval()?),
                MinMaxOp::Min => Distribution::min(&result, &term.eval()?),
            };
        }
        Ok(result)
    }
}

#[derive(Debug)]
enum MinMaxOp {
    Min,
    Max,
}
impl<'pest> FromPest<'pest> for MinMaxOp {
    type FatalError = Void;
    type Rule = Rule;

    fn from_pest(
        pest: &mut Pairs<'pest, Self::Rule>,
    ) -> Result<Self, ConversionError<Self::FatalError>> {
        match pest.peek() {
            Some(pair) => {
                match pair.as_rule() {
                    Rule::MinMaxOp => {
                        match pest.next().unwrap().as_str() {
                            "^" => Ok(MinMaxOp::Max),
                            "v" => Ok(MinMaxOp::Min),
                            _ => unreachable!("No other strings matched by rule"),
                        }
                    },
                    _ => Err(ConversionError::NoMatch),
                }
            },
            None => Err(ConversionError::NoMatch),
        }
    }
}

#[derive(FromPest, Debug)]
#[pest_ast(rule(Rule::AddTerm))]
struct AddTerm {
    sign: Option<AddOp>,
    main_term: MulTerm,
    additions: Vec<(AddOp, MulTerm)>,
}
impl AddTerm {
    fn eval(&self) -> Result<Distribution, Error> {
        let mut result = self.main_term.eval()?;
        if let Some(AddOp::Minus) = self.sign {
            result = -result;
        }
        for (op, term) in &self.additions {
            result = match op {
                AddOp::Plus => result + term.eval()?,
                AddOp::Minus => result - term.eval()?,
            };
        }
        Ok(result)
    }
}

#[derive(Debug)]
enum AddOp {
    Plus,
    Minus,
}
impl<'pest> FromPest<'pest> for AddOp {
    type FatalError = Void;
    type Rule = Rule;

    fn from_pest(
        pest: &mut Pairs<'pest, Self::Rule>,
    ) -> Result<Self, ConversionError<Self::FatalError>> {
        match pest.peek() {
            Some(pair) => {
                match pair.as_rule() {
                    Rule::AddOp => {
                        match pest.next().unwrap().as_str() {
                            "+" => Ok(AddOp::Plus),
                            "-" => Ok(AddOp::Minus),
                            _ => unreachable!("No other strings matched by rule"),
                        }
                    },
                    _ => Err(ConversionError::NoMatch),
                }
            },
            None => Err(ConversionError::NoMatch),
        }
    }
}

#[derive(FromPest, Debug)]
#[pest_ast(rule(Rule::MulTerm))]
struct MulTerm {
    main_term: Repeat,
    additions: Vec<(MulOp, Repeat)>,
}
impl MulTerm {
    fn eval(&self) -> Result<Distribution, Error> {
        let mut result = self.main_term.eval()?;
        for (op, term) in &self.additions {
            result = match op {
                MulOp::Multiply => result * term.eval()?,
                MulOp::TrueDivide => result.true_div(&term.eval()?)?,
                MulOp::TruncDivide => result.trunc_div(&term.eval()?)?,
                MulOp::FloorDivide => result.floor_div(&term.eval()?)?,
                MulOp::CeilDivide => result.ceil_div(&term.eval()?)?,
                MulOp::Modulo => result.modulo(&term.eval()?)?,
            };
        }
        Ok(result)
    }
}

#[derive(Debug)]
enum MulOp {
    Multiply,
    TrueDivide,
    TruncDivide,
    FloorDivide,
    CeilDivide,
    Modulo,
}
impl<'pest> FromPest<'pest> for MulOp {
    type FatalError = Void;
    type Rule = Rule;

    fn from_pest(
        pest: &mut Pairs<'pest, Self::Rule>,
    ) -> Result<Self, ConversionError<Self::FatalError>> {
        match pest.peek() {
            Some(pair) => {
                match pair.as_rule() {
                    Rule::MulOp => {
                        match pest.next().unwrap().as_str() {
                            "*" => Ok(MulOp::Multiply),
                            "/" => Ok(MulOp::TrueDivide),
                            "//" => Ok(MulOp::TruncDivide),
                            "/v" => Ok(MulOp::FloorDivide),
                            "/^" => Ok(MulOp::CeilDivide),
                            "%" => Ok(MulOp::Modulo),
                            _ => unreachable!("No other strings matched by rule"),
                        }
                    },
                    _ => Err(ConversionError::NoMatch),
                }
            },
            None => Err(ConversionError::NoMatch),
        }
    }
}

#[derive(FromPest, Debug)]
#[pest_ast(rule(Rule::Repeat))]
struct Repeat {
    main_term: Atom,
    repetitions: Vec<(Option<Keep>, Atom)>,
}
impl Repeat {
    fn eval(&self) -> Result<Distribution, Error> {
        let mut result = self.main_term.eval()?;
        for (keep, term) in &self.repetitions {
            result = match keep {
                Some(Keep {
                    kind: LowHigh::Low,
                    val,
                }) => term.eval()?.repeat_disadvantage(&result, &val.eval()?)?,
                Some(Keep {
                    kind: LowHigh::High,
                    val,
                }) => term.eval()?.repeat_advantage(&result, &val.eval()?)?,
                None => term.eval()?.repeat(&result)?,
            }
        }
        Ok(result)
    }
}

#[derive(FromPest, Debug)]
#[pest_ast(rule(Rule::Keep))]
struct Keep {
    kind: LowHigh,
    val: Atom,
}

#[derive(Debug)]
enum LowHigh {
    Low,
    High,
}
impl<'pest> FromPest<'pest> for LowHigh {
    type FatalError = Void;
    type Rule = Rule;

    fn from_pest(
        pest: &mut Pairs<'pest, Self::Rule>,
    ) -> Result<Self, ConversionError<Self::FatalError>> {
        match pest.peek() {
            Some(pair) => {
                match pair.as_rule() {
                    Rule::LowHigh => {
                        match pest.next().unwrap().as_str() {
                            "l" => Ok(LowHigh::Low),
                            "h" | "" => Ok(LowHigh::High),
                            _ => unreachable!("No other strings matched by rule"),
                        }
                    },
                    _ => Err(ConversionError::NoMatch),
                }
            },
            None => Err(ConversionError::NoMatch),
        }
    }
}

#[derive(Debug)]
enum Atom {
    Literal(Number),
    Expr(Box<Expr>),
    DieRoll(Box<DieRoll>),
    VariadicCall(VariadicCall),
    UnaryCall(UnaryCall),
}
impl<'pest> FromPest<'pest> for Atom {
    type FatalError = Void;
    type Rule = Rule;

    fn from_pest(
        pest: &mut Pairs<'pest, Self::Rule>,
    ) -> Result<Self, ConversionError<Self::FatalError>> {
        match pest.peek() {
            Some(pair) => {
                match pair.as_rule() {
                    Rule::Number => Ok(Atom::Literal(Number::from_pest(pest)?)),
                    Rule::Expression => {
                        Ok(Atom::Expr(Box::new(Expr::from_pest(pest)?)))
                    },
                    Rule::DieRoll => {
                        Ok(Atom::DieRoll(Box::new(DieRoll::from_pest(pest)?)))
                    },
                    Rule::VariadicCall => {
                        Ok(Atom::VariadicCall(VariadicCall::from_pest(pest)?))
                    },
                    Rule::UnaryCall => Ok(Atom::UnaryCall(UnaryCall::from_pest(pest)?)),
                    _ => Err(ConversionError::NoMatch),
                }
            },
            None => Err(ConversionError::NoMatch),
        }
    }
}
impl Atom {
    fn eval(&self) -> Result<Distribution, Error> {
        match self {
            Atom::Literal(n) => Ok(n.eval()),
            Atom::Expr(e) => e.eval(),
            Atom::DieRoll(d) => d.eval(),
            Atom::VariadicCall(f) => f.eval(),
            Atom::UnaryCall(f) => f.eval(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum FunctionCallError {
    NotEnoughArgs { expected: usize, got: usize },
    TooManyArgs { expected: usize, got: usize },
}

#[derive(FromPest, Debug)]
#[pest_ast(rule(Rule::VariadicCall))]
struct VariadicCall {
    name: VariadicFnName,
    first_arg: Box<Expr>,
    args: Vec<Expr>,
}
impl VariadicCall {
    fn eval(&self) -> Result<Distribution, Error> {
        match self.name {
            VariadicFnName::Min => {
                let mut result = self.first_arg.eval()?;
                for arg in &self.args {
                    result = Distribution::min(&result, &arg.eval()?);
                }
                result.simplify();
                Ok(result)
            },
            VariadicFnName::Max => {
                let mut result = self.first_arg.eval()?;
                for arg in &self.args {
                    result = Distribution::max(&result, &arg.eval()?);
                }
                result.simplify();
                Ok(result)
            },
            #[allow(clippy::cast_possible_wrap)]
            VariadicFnName::Avg => {
                let mut result = self.first_arg.eval()?;
                for arg in &self.args {
                    result += arg.eval()?;
                }
                result =
                    result.true_div(&Distribution::from(self.args.len() as isize))?;
                result.simplify();
                Ok(result)
            },
            VariadicFnName::Equal => {
                let mut total_true_cases = 0;
                let first_arg = self.first_arg.eval()?;
                let args = self
                    .args
                    .iter()
                    .map(Expr::eval)
                    .collect::<Result<Vec<_>, _>>()?;
                let total_cases = first_arg.count
                    * args.iter().map(|arg| arg.count).product::<usize>();
                for (value, count) in first_arg.contents {
                    let mut true_cases = count;
                    for arg in &args {
                        if let Some(count) = arg.contents.get(&value) {
                            true_cases *= count;
                        } else {
                            true_cases = 0;
                            break;
                        }
                    }
                    total_true_cases += true_cases;
                }
                Ok(Distribution::new(
                    [
                        (Value::new(0, 1), total_cases - total_true_cases),
                        (Value::new(1, 1), total_true_cases),
                    ]
                    .into(),
                    total_cases,
                ))
            },
            VariadicFnName::Sum => {
                let mut result = self.first_arg.eval()?;
                for arg in &self.args {
                    result += arg.eval()?;
                }
                result.simplify();
                Ok(result)
            },
            VariadicFnName::Pow => {
                if self.args.is_empty() {
                    Err(Error::FunctionCallError(FunctionCallError::NotEnoughArgs {
                        expected: 2,
                        got: 1,
                    }))
                } else if self.args.len() > 1 {
                    Err(Error::FunctionCallError(FunctionCallError::TooManyArgs {
                        expected: 2,
                        got: self.args.len() + 1,
                    }))
                } else {
                    let mut result = self.first_arg.eval()?;
                    result = result.pow(&self.args[0].eval()?)?;
                    result.simplify();
                    Ok(result)
                }
            },
        }
    }
}

#[derive(Debug)]
enum VariadicFnName {
    Min,
    Max,
    Avg,
    Equal,
    Sum,
    Pow,
}
impl<'pest> FromPest<'pest> for VariadicFnName {
    type FatalError = Void;
    type Rule = Rule;

    fn from_pest(
        pest: &mut Pairs<'pest, Self::Rule>,
    ) -> Result<Self, ConversionError<Self::FatalError>> {
        match pest.peek() {
            Some(pair) => {
                match pair.as_rule() {
                    Rule::VariadicFnName => {
                        match pest.next().unwrap().as_str() {
                            "min" => Ok(Self::Min),
                            "max" => Ok(Self::Max),
                            "avg" => Ok(Self::Avg),
                            "equal" => Ok(Self::Equal),
                            "sum" => Ok(Self::Sum),
                            "pow" => Ok(Self::Pow),
                            _ => unreachable!("No other strings matched by rule"),
                        }
                    },
                    _ => Err(ConversionError::NoMatch),
                }
            },
            None => Err(ConversionError::NoMatch),
        }
    }
}

#[derive(FromPest, Debug)]
#[pest_ast(rule(Rule::UnaryCall))]
struct UnaryCall {
    name: UnaryFnName,
    arg: Box<Expr>,
}
impl UnaryCall {
    fn eval(&self) -> Result<Distribution, Error> {
        match self.name {
            UnaryFnName::Sqrt => Ok(self.arg.eval()?.sqrt()?),
            UnaryFnName::Floor => Ok(self.arg.eval()?.floor()),
            UnaryFnName::Ceil => Ok(self.arg.eval()?.ceil()),
            UnaryFnName::Round => Ok(self.arg.eval()?.round()),
            UnaryFnName::Trunc => Ok(self.arg.eval()?.trunc()),
            UnaryFnName::Abs => Ok(self.arg.eval()?.abs()),
        }
    }
}

#[derive(Debug)]
enum UnaryFnName {
    Sqrt,
    Floor,
    Ceil,
    Round,
    Trunc,
    Abs,
}
impl<'pest> FromPest<'pest> for UnaryFnName {
    type FatalError = Void;
    type Rule = Rule;

    fn from_pest(
        pest: &mut Pairs<'pest, Self::Rule>,
    ) -> Result<Self, ConversionError<Self::FatalError>> {
        match pest.peek() {
            Some(pair) => {
                match pair.as_rule() {
                    Rule::UnaryFnName => {
                        match pest.next().unwrap().as_str() {
                            "sqrt" => Ok(Self::Sqrt),
                            "floor" => Ok(Self::Floor),
                            "ceil" => Ok(Self::Ceil),
                            "round" => Ok(Self::Round),
                            "trunc" => Ok(Self::Trunc),
                            "abs" => Ok(Self::Abs),
                            _ => unreachable!("No other strings matched by rule"),
                        }
                    },
                    _ => Err(ConversionError::NoMatch),
                }
            },
            None => Err(ConversionError::NoMatch),
        }
    }
}

#[derive(Debug)]
struct Number(Value);
impl<'pest> FromPest<'pest> for Number {
    type FatalError = Void;
    type Rule = Rule;

    fn from_pest(
        pest: &mut Pairs<'pest, Self::Rule>,
    ) -> Result<Self, ConversionError<Self::FatalError>> {
        match pest.peek() {
            Some(pair) => {
                match pair.as_rule() {
                    Rule::Number => {
                        let decimal_repr = pest.next().unwrap().as_str();
                        if let Some((whole, frac)) = decimal_repr.split_once('.') {
                            let whole = whole.parse::<isize>().unwrap();
                            #[allow(clippy::cast_possible_truncation)]
                            let denom = 10_isize.pow(frac.len() as u32);
                            let frac = frac.parse::<isize>().unwrap();
                            Ok(Number(Value::new(whole * denom + frac, denom)))
                        } else {
                            let whole = decimal_repr.parse::<isize>().unwrap();
                            Ok(Number(Value::new(whole, 1)))
                        }
                    },
                    _ => Err(ConversionError::NoMatch),
                }
            },
            None => Err(ConversionError::NoMatch),
        }
    }
}
impl Number {
    fn eval(&self) -> Distribution {
        Distribution::from(self.0)
    }
}

#[derive(FromPest, Debug)]
#[pest_ast(rule(Rule::DieRoll))]
struct DieRoll {
    num_sides: Atom,
}
impl DieRoll {
    fn eval(&self) -> Result<Distribution, Error> {
        Ok(Distribution::uniform_over(&self.num_sides.eval()?)?)
    }
}
