use crate::prelude::*;

// ILS
pub struct Program {
    pub moves: Vec<Move>,
}

impl std::fmt::Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for m in &self.moves {
            writeln!(f, "{}", m)?;
        }
        Ok(())
    }
}

pub type Block = Vec<BlockId>;

struct IslPoint<'a>(&'a Point);

impl std::fmt::Display for IslPoint<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{},{}]", self.0 .0, self.0 .1)
    }
}

struct IslBlock<'a>(&'a Block);

impl std::fmt::Display for IslBlock<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        assert!(!self.0.is_empty());
        write!(f, "[")?;
        write!(f, "{}", self.0[0])?;
        for block_id in self.0.iter().skip(1) {
            write!(f, ".{}", block_id)?;
        }
        write!(f, "]")
    }
}

#[test]
fn isl_display() {
    let block = vec![1, 2, 2];
    assert_eq!(IslBlock(&block).to_string(), "[1.2.2]");

    let pcut = Move::PCut {
        block: block.clone(),
        point: (1, 2),
    };
    assert_eq!(pcut.to_string(), "cut [1.2.2] [1,2]");

    let lcut = Move::LCut {
        block: block.clone(),
        orientation: Orientation::X,
        line_number: 3,
    };
    assert_eq!(lcut.to_string(), "cut [1.2.2] [X] [3]");

    let color = Move::Color {
        block: block.clone(),
        color: [1, 2, 3, 4],
    };
    assert_eq!(color.to_string(), "color [1.2.2] [1,2,3,4]");

    let block2 = vec![2];

    let swap = Move::Swap {
        block1: block.clone(),
        block2: block2.clone(),
    };
    assert_eq!(swap.to_string(), "swap [1.2.2] [2]");

    let merge = Move::Merge {
        block1: block.clone(),
        block2: block2.clone(),
    };
    assert_eq!(merge.to_string(), "merge [1.2.2] [2]");
}

// use derive_more::{Display};

#[derive(Copy, Clone)]
pub enum Orientation {
    X,
    Y,
}

struct IslOrientation<'a>(&'a Orientation);

impl std::fmt::Display for IslOrientation<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}]",
            match &self.0 {
                Orientation::X => "X",
                Orientation::Y => "Y",
            }
        )
    }
}

struct IslRgba<'a>(&'a Rgba);

impl std::fmt::Display for IslRgba<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{},{},{},{}]",
            self.0[0], self.0[1], self.0[2], self.0[3]
        )
    }
}

#[derive(Clone)]
pub enum Move {
    PCut {
        block: Block,
        point: Point,
    },
    LCut {
        block: Block,
        orientation: Orientation,
        line_number: Coord,
    },
    Color {
        block: Block,
        color: Rgba,
    },
    Swap {
        block1: Block,
        block2: Block,
    },
    Merge {
        block1: Block,
        block2: Block,
    },
}

impl std::fmt::Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Move::*;
        match self {
            PCut { block, point } => {
                write!(f, "cut {} {}", IslBlock(block), IslPoint(point))
            }
            LCut {
                block,
                orientation,
                line_number,
            } => {
                write!(
                    f,
                    "cut {} {} [{}]",
                    IslBlock(block),
                    IslOrientation(orientation),
                    line_number
                )
            }
            Color { block, color } => {
                write!(f, "color {} {}", IslBlock(block), IslRgba(color),)
            }
            Swap { block1, block2 } => {
                write!(f, "swap {} {}", IslBlock(block1), IslBlock(block2),)
            }
            Merge { block1, block2 } => {
                write!(f, "merge {} {}", IslBlock(block1), IslBlock(block2),)
            }
        }
    }
}
