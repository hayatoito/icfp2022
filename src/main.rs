mod prelude;

pub mod ils;
pub mod solver;
pub mod submission;

use clap::Parser;

use crate::prelude::*;

#[derive(Parser, Debug)]
#[clap(name = "my-icfp2022")]
enum Cli {
    DumpProblem { id: ProblemId },
    Solve { id: ProblemId },
}

fn main() -> Result<()> {
    env_logger::init();
    match Cli::from_args() {
        Cli::DumpProblem { id } => {
            crate::solver::dump_problem(id)?;
        }
        Cli::Solve { id } => {
            crate::solver::solve(id)?;
        }
    }
    Ok(())
}
