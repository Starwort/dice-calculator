use std::env::args;

fn main() {
    let input = args().nth(1).expect("missing input");
    let result = dice::parse(&input).unwrap();
    let mut result = result.eval().unwrap();
    result.simplify();
    // let (result, total) = result.into_raw();
    let result = result.into_approximate();
    for (result, chance) in result {
        println!("{:>8}: {:>10.7}%", result, chance * 100.0);
    }
}
