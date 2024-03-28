use std::env::args;

use itertools::Itertools;
use lowcharts::plot;

fn main() {
    let input = args().nth(1).expect("missing input");
    let result = dice::parse(&input).unwrap();
    let mut result = result.eval().unwrap();
    result.simplify();
    let (result, _) = dbg!(result).into_raw();
    println!(
        "{}",
        plot::MatchBar::new(
            result
                .into_iter()
                .map(|(k, v)| {
                    plot::MatchBarRow {
                        label: format!("{k}"),
                        count: v,
                    }
                })
                .collect_vec()
        )
    );
}
