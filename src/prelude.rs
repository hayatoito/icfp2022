pub use anyhow::{bail, ensure, Context, Result};
pub use log::*;
pub use serde::{Deserialize, Serialize};
pub use std::collections::{HashMap, HashSet, VecDeque};
pub use std::io::Write;
pub use std::ops::Deref;
use std::ops::Index;
use std::ops::IndexMut;
pub use std::ops::Range;
pub use std::path::{Path, PathBuf};

use image::GenericImageView;

pub fn read_from_task_dir(task_relative_path: impl AsRef<Path>) -> Result<String> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("task");
    path.push(task_relative_path);
    Ok(std::fs::read_to_string(path)?)
}

pub fn write_to_task_dir(task_relative_path: impl AsRef<Path>, content: &str) -> Result<()> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("task");
    path.push(task_relative_path);
    Ok(std::fs::write(path, content)?)
}

pub type Cost = i64;

pub type ProblemId = u64;

// const STEP_BY: usize = 15;
// const STEP_BY: usize = 5;
// const STEP_BY: usize = 40;
const STEP_BY: usize = 20;

#[derive(Copy, Clone, derive_more::Display)]
pub enum SpecVersion {
    V0,
    V1,
    V2,
}

pub const COLOR_COST: Cost = 5;
#[allow(dead_code)]
pub const SWAP_COST: Cost = 3;
pub const MERGE_COST: Cost = 1;

impl SpecVersion {
    pub fn line_cut_cost(&self) -> Cost {
        use SpecVersion::*;
        match self {
            V0 | V1 => 7,
            V2 => 2,
        }
    }

    pub fn point_cut_cost(&self) -> Cost {
        use SpecVersion::*;
        match self {
            V0 | V1 => 10,
            V2 => 3,
        }
    }
}

pub struct Problem {
    pub spec_version: SpecVersion,
    pub painting: Painting,
    pub initial_blocks: Vec<InitialBlock>,
    pub preprocess_moves_cost: Cost,
    pub preprocess_moves: Vec<crate::ils::Move>,
}

impl Problem {
    pub fn new(id: ProblemId) -> Result<Problem> {
        let spec_version = match id {
            0 => unreachable!(),
            1..=25 => SpecVersion::V0,
            26..=35 => SpecVersion::V1,
            36.. => SpecVersion::V2,
        };

        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("task");
        path.push(&format!("problem/{}.png", id));

        let img = image::open(&path).context(format!("path is not found?: {:?}", &path))?;
        debug!("id: {id}, dimensions {:?}", img.dimensions());
        debug!("{:?}", img.color());

        let mut pixels = vec![WHITE; (img.width() * img.height()) as usize];
        for (x, y, p) in img.pixels() {
            // Reverse vertically to align coord system.
            pixels[((img.height() - 1 - y) * img.width() + x) as usize] = [p[0], p[1], p[2], p[3]];
        }

        let painting = Painting {
            width: img.width(),
            height: img.height(),
            pixels,
        };

        debug!("painting[(5,5)] = {:?}", painting[(5, 5)]);

        let mut initial_blocks = vec![];

        let mut preprocess_move_cost = 0;
        let mut swap_merge_and_color_moves = vec![];

        match spec_version {
            SpecVersion::V0 => {
                initial_blocks.push(InitialBlock {
                    id: 0,
                    shape: painting.shape(),
                    color: WHITE,
                });
            }
            SpecVersion::V1 => {
                let initial_config = InitialConfig::new(id)?;
                assert_eq!(initial_config.width, img.width());
                assert_eq!(initial_config.height, img.height());
                for block in &initial_config.blocks {
                    initial_blocks.push(InitialBlock {
                        id: block.block_id.parse()?,
                        shape: Shape {
                            x: block.bottoml_left[0],
                            y: block.bottoml_left[1],
                            width: block.top_right[0] - block.bottoml_left[0],
                            height: block.top_right[1] - block.bottoml_left[1],
                        },
                        // V1 initial block always has color.
                        color: block.color.unwrap(),
                    });
                }
            }
            SpecVersion::V2 => {
                let initial_config = InitialConfig::new(id)?;
                assert_eq!(initial_config.width, img.width());
                assert_eq!(initial_config.height, img.height());
                for block in &initial_config.blocks {
                    let id = block.block_id.parse()?;
                    let shape = Shape {
                        x: block.bottoml_left[0],
                        y: block.bottoml_left[1],
                        width: block.top_right[0] - block.bottoml_left[0],
                        height: block.top_right[1] - block.bottoml_left[1],
                    };

                    // TODO: Support v2 png.
                    // For a while, color it at first.
                    let color = painting.most_used_color(&shape).unwrap();
                    preprocess_move_cost += COLOR_COST;
                    swap_merge_and_color_moves.push(crate::ils::Move::Color {
                        block: vec![id],
                        color,
                    });

                    initial_blocks.push(InitialBlock { id, shape, color });
                }
            }
        }

        Ok(Problem {
            spec_version,
            painting,
            initial_blocks,
            preprocess_moves_cost: preprocess_move_cost,
            preprocess_moves: swap_merge_and_color_moves,
        })
    }

    pub fn merge_initial_blocks(&self) -> Problem {
        let mut shape_merger = ShapeMerger {
            shapes: self
                .initial_blocks
                .iter()
                .map(|b| (b.id, b.shape))
                .collect(),
        };
        let (mut moves, merge_cost, new_shapes) = shape_merger.merge(self.painting.size());

        let mut color_cost = 0;

        let mut initial_blocks = vec![];

        for (id, shape) in new_shapes {
            let color = self.painting.most_used_color(&shape).unwrap();
            moves.push(crate::ils::Move::Color {
                block: vec![id],
                color,
            });
            color_cost += (COLOR_COST as f64 * (self.painting.size() as f64)
                / (shape.size() as f64))
                .round() as Cost;
            initial_blocks.push(InitialBlock { id, shape, color });
        }

        Problem {
            spec_version: self.spec_version,
            painting: self.painting.clone(),
            initial_blocks,
            preprocess_moves_cost: merge_cost + color_cost,
            preprocess_moves: moves,
        }
    }
}

pub type BlockId = u32;

pub type Coord = u32;
pub type Point = (Coord, Coord);

pub type ColorUnit = u8;
pub type Rgba = [ColorUnit; 4];
pub const WHITE: Rgba = [255, 255, 255, 255];

trait PixelDiff {
    fn pixel_diff(&self, other: &Rgba) -> f64;
}

fn square_dist(c1: ColorUnit, c2: ColorUnit) -> u64 {
    let diff = c1.abs_diff(c2) as u64;
    diff * diff
}

impl PixelDiff for Rgba {
    fn pixel_diff(&self, other: &Rgba) -> f64 {
        let diff_sum: u64 = self
            .iter()
            .zip(other.iter())
            .map(|(c1, c2)| square_dist(*c1, *c2))
            .sum();
        (diff_sum as f64).sqrt()
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug, Serialize, Deserialize)]
pub struct Shape {
    pub x: Coord,
    pub y: Coord,
    pub width: Coord,
    pub height: Coord,
}

impl Shape {
    pub fn size(&self) -> u32 {
        self.width * self.height
    }

    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    pub fn range_x(&self) -> Range<Coord> {
        self.x..self.x + self.width
    }

    pub fn range_y(&self) -> Range<Coord> {
        self.y..self.y + self.height
    }

    pub fn merge(&self, other: &Shape) -> Option<Shape> {
        if self.width == other.width && self.x == other.x {
            // Try to merge vertically
            if self.y + self.height == other.y {
                return Some(Shape {
                    x: self.x,
                    y: self.y,
                    width: self.width,
                    height: self.height + other.height,
                });
            }
            if other.y + other.height == self.y {
                return Some(Shape {
                    x: other.x,
                    y: other.y,
                    width: self.width,
                    height: self.height + other.height,
                });
            }
        }
        if self.height == other.height && self.y == other.y {
            // Try to merge horizontally
            if self.x + self.width == other.x {
                return Some(Shape {
                    x: self.x,
                    y: self.y,
                    width: self.width + other.width,
                    height: self.height,
                });
            }
            if other.x + other.width == self.x {
                return Some(Shape {
                    x: other.x,
                    y: other.y,
                    width: self.width + other.width,
                    height: self.height,
                });
            }
        }
        None
    }

    fn inside(&self, p: &Point) -> bool {
        self.x <= p.0 && p.0 < self.x + self.width && self.y <= p.1 && p.1 < self.y + self.height
    }

    pub fn interesting_pcut(&self, interesting_point: &[Point]) -> Vec<Vec<Shape>> {
        interesting_point
            .iter()
            .filter(|p| {
                self.inside(p)
                    && p.0 != self.x
                    && p.0 != self.x + self.width - 1
                    && p.1 != self.y
                    && p.1 != self.y + self.height - 1
            })
            .map(|p| self.pcut(p.0, p.1))
            .collect()
    }

    pub fn interesting_lcutx(&self, interesting_line: &[Coord]) -> Vec<Vec<Shape>> {
        interesting_line
            .iter()
            .filter(|x| self.x < **x && **x < self.x + self.width - 1)
            .map(|x| self.lcut_x(*x))
            .collect()
    }

    pub fn interesting_lcuty(&self, interesting_line: &[Coord]) -> Vec<Vec<Shape>> {
        interesting_line
            .iter()
            .filter(|y| self.y < **y && **y < self.y + self.height - 1)
            .map(|y| self.lcut_y(*y))
            .collect()
    }

    pub fn all_pcut(&self) -> Vec<Vec<Shape>> {
        assert!(!self.is_empty());
        let mut all = vec![];
        for mid_y in self.range_y().step_by(STEP_BY) {
            if mid_y == self.y || mid_y == self.y + self.height - 1 {
                continue;
            }
            for mid_x in self.range_x().step_by(STEP_BY) {
                if mid_x == self.x || mid_x == self.x + self.width - 1 {
                    continue;
                }
                all.push(self.pcut(mid_x, mid_y))
            }
        }
        all
    }

    fn pcut(&self, mid_x: Coord, mid_y: Coord) -> Vec<Shape> {
        assert!(!self.is_empty());

        let left_width = mid_x - self.x;
        let right_width = self.width - left_width;

        let bottom_height = mid_y - self.y;
        let top_height = self.height - bottom_height;

        vec![
            // 0
            Shape {
                x: self.x,
                y: self.y,
                width: left_width,
                height: bottom_height,
            },
            // 1
            Shape {
                x: mid_x,
                y: self.y,
                width: right_width,
                height: bottom_height,
            },
            // 2
            Shape {
                x: mid_x,
                y: mid_y,
                width: right_width,
                height: top_height,
            },
            // 3
            Shape {
                x: self.x,
                y: mid_y,
                width: left_width,
                height: top_height,
            },
        ]
    }

    pub fn all_lcut_x(&self) -> Vec<Vec<Shape>> {
        assert!(!self.is_empty());
        let mut all = vec![];
        for mid_x in self.range_x().step_by(STEP_BY) {
            if mid_x == self.x || mid_x == self.x + self.width - 1 {
                continue;
            }
            all.push(self.lcut_x(mid_x))
        }
        all
    }

    fn lcut_x(&self, mid_x: Coord) -> Vec<Shape> {
        assert!(!self.is_empty());
        assert!(self.width > 1);

        vec![
            // 0
            Shape {
                x: self.x,
                y: self.y,
                width: mid_x - self.x,
                height: self.height,
            },
            // 1
            Shape {
                x: mid_x,
                y: self.y,
                width: self.x + self.width - mid_x,
                height: self.height,
            },
        ]
    }

    pub fn all_lcut_y(&self) -> Vec<Vec<Shape>> {
        assert!(!self.is_empty());
        let mut all = vec![];
        for mid_y in self.range_y().step_by(STEP_BY) {
            if mid_y == self.y || mid_y == self.y + self.height - 1 {
                continue;
            }
            all.push(self.lcut_y(mid_y))
        }
        all
    }

    fn lcut_y(&self, mid_y: Coord) -> Vec<Shape> {
        assert!(!self.is_empty());
        assert!(self.height > 1);

        vec![
            // 0
            Shape {
                x: self.x,
                y: self.y,
                width: self.width,
                height: mid_y - self.y,
            },
            // 1
            Shape {
                x: self.x,
                y: mid_y,
                width: self.width,
                height: self.y + self.height - mid_y,
            },
        ]
    }
}

pub struct ShapeMerger {
    pub shapes: HashMap<BlockId, Shape>,
}

impl ShapeMerger {
    fn find(&self) -> Option<(BlockId, BlockId, u32, Shape)> {
        let mut best = 10000000;
        // Prefer small merge
        let mut best_merge = None;

        for (id1, block1) in &self.shapes {
            for (id2, block2) in &self.shapes {
                if id1 == id2 {
                    continue;
                }
                if let Some(shape) = block1.merge(block2) {
                    let larger = block1.size().max(block2.size());
                    if best > larger {
                        best = larger;
                        best_merge = Some((*id1, *id2, larger, shape));
                    }
                }
            }
        }
        best_merge
    }

    pub fn merge(
        &mut self,
        canvas_size: Coord,
    ) -> (Vec<crate::ils::Move>, Cost, HashMap<BlockId, Shape>) {
        info!("shapes.len(): {}", self.shapes.len());

        let mut moves = vec![];
        let mut cost = 0;

        let mut next_block_id = self.shapes.len() as BlockId;

        while let Some((block1, block2, larger_size, shape)) = self.find() {
            self.shapes.remove(&block1);
            self.shapes.remove(&block2);
            moves.push(crate::ils::Move::Merge {
                block1: vec![block1],
                block2: vec![block2],
            });
            cost += (MERGE_COST as f64 * canvas_size as f64 / larger_size as f64).round() as Cost;
            self.shapes.insert(next_block_id, shape);
            next_block_id += 1;
        }
        if self.shapes.len() != 1 {
            warn!("merged shapes' len(): {}", self.shapes.len());
        }

        (moves, cost, self.shapes.clone())
    }
}

// Painting
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Painting {
    pub width: Coord,
    pub height: Coord,
    pub pixels: Vec<Rgba>,
}

impl Painting {
    pub fn size(&self) -> u32 {
        self.width * self.height
    }

    pub fn shape(&self) -> Shape {
        Shape {
            x: 0,
            y: 0,
            width: self.width,
            height: self.height,
        }
    }

    pub fn save_as(&self, name: &str) -> Result<()> {
        debug!(
            "saving painting: width: {}, height: {}, size: {}",
            self.width,
            self.height,
            self.pixels.len()
        );
        let img = image::ImageBuffer::from_fn(self.width, self.height, |x, y| {
            assert!(x < self.width);
            assert!(y < self.height);
            image::Rgba(self[(x, self.height - 1 - y)])
            // image::Rgba(self[(x, y)])
        });

        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("task");
        path.push("paint");
        path.push(name);

        img.save(path)?;
        Ok(())
    }

    pub fn most_used_color(&self, shape: &Shape) -> Option<Rgba> {
        let mut freq = HashMap::<Rgba, u32>::new();

        for y in shape.range_y() {
            for x in shape.range_x() {
                *freq.entry(self[(x, y)]).or_insert(0) += 1;
            }
        }
        freq.into_iter().max_by_key(|&(_k, v)| v).map(|(k, _v)| k)
    }

    pub fn used_colors(&self, shape: &Shape) -> Vec<Rgba> {
        let mut set = HashSet::<Rgba>::new();

        for y in shape.range_y() {
            for x in shape.range_x() {
                set.insert(self[(x, y)]);
            }
        }
        set.into_iter().collect()
    }

    pub fn similarity(&self, other: &Painting) -> Cost {
        let pixel_diff_sum: f64 = self
            .pixels
            .iter()
            .zip(other.pixels.iter())
            .map(|(p1, p2)| p1.pixel_diff(p2))
            .sum();
        (pixel_diff_sum * 0.005).round() as Cost
    }

    /// color -> [index(x, y)] -> similarity in shape(0, 0, x, y) for color
    pub fn build_similarity_table(&self, colors: &HashSet<Rgba>) -> HashMap<Rgba, Vec<f64>> {
        let mut color_table = HashMap::<Rgba, Vec<f64>>::new();

        let table_index = |x: Coord, y: Coord| (y * (self.width + 1) + x) as usize;

        for color in colors {
            // Prepare +1 size table.
            let mut sim_table = vec![0.0; ((self.width + 1) * (self.height + 1)) as usize];

            for y in 0..self.height + 1 {
                for x in 0..self.width + 1 {
                    if x == 0 || y == 0 {
                        continue;
                    }
                    let i = table_index(x, y);
                    let this_pixel_diff = self[(x - 1, y - 1)].pixel_diff(color);
                    match (x, y) {
                        (1, 1) => sim_table[i] = this_pixel_diff,
                        (x, 1) => sim_table[i] = this_pixel_diff + sim_table[table_index(x - 1, 1)],
                        (1, y) => sim_table[i] = this_pixel_diff + sim_table[table_index(1, y - 1)],
                        (x, y) => {
                            sim_table[i] = this_pixel_diff
                                + sim_table[table_index(x, y - 1)]
                                + sim_table[table_index(x - 1, y)]
                                - sim_table[table_index(x - 1, y - 1)]
                        }
                    }
                }
            }
            // info!(
            //     "build similarity_table: color: {:?}: table.len: {}, sum: {}",
            //     color,
            //     sim_table.len(),
            //     sim_table.iter().sum::<f64>()
            // );
            color_table.insert(*color, sim_table);
        }
        debug!("build similarity_table: colors: {}", color_table.len());
        color_table
    }

    pub fn similarity_in_shape_with_color(&self, shape: &Shape, color: &Rgba) -> Cost {
        let mut diff_sum = 0.0;
        for y in shape.range_y() {
            for x in shape.range_x() {
                diff_sum += self[(x, y)].pixel_diff(color)
            }
        }
        (diff_sum * 0.005).round() as Cost
    }

    pub fn same_color(&self, p1: Point, p2: Point) -> bool {
        self[p1] == self[p2]
    }

    pub fn interesting_pcut_point(&self) -> Vec<Point> {
        let mut ps = HashSet::new();
        for y in 1..self.height - 1 {
            for x in 1..self.width - 1 {
                //   o
                // x P o
                //   x
                if self.same_color((x, y), (x + 1, y))
                    && self.same_color((x, y), (x, y + 1))
                    && !self.same_color((x, y), (x - 1, y))
                    && !self.same_color((x, y), (x, y - 1))
                {
                    ps.insert((x, y));
                }
                //   P
                // x o o
                //   o
                if self.same_color((x, y), (x + 1, y))
                    && self.same_color((x, y), (x, y - 1))
                    && !self.same_color((x, y), (x - 1, y))
                    && !self.same_color((x, y), (x, y + 1))
                {
                    ps.insert((x, y + 1));
                }
                //   x P
                // o o x
                //   o
                if self.same_color((x, y), (x - 1, y))
                    && self.same_color((x, y), (x, y - 1))
                    && !self.same_color((x, y), (x + 1, y))
                    && !self.same_color((x, y), (x, y + 1))
                {
                    ps.insert((x + 1, y + 1));
                }
                //   o
                // o o P
                //   x
                if self.same_color((x, y), (x - 1, y))
                    && self.same_color((x, y), (x, y + 1))
                    && !self.same_color((x, y), (x + 1, y))
                    && !self.same_color((x, y), (x, y - 1))
                {
                    ps.insert((x + 1, y));
                }
            }
        }
        ps.into_iter().collect()
    }

    pub fn interesting_lcut_lines(&self) -> (Vec<Coord>, Vec<Coord>) {
        let mut xs = HashSet::new();
        let mut ys = HashSet::new();
        for y in 1..self.height - 1 {
            for x in 1..self.width - 1 {
                // x P
                if !self.same_color((x, y), (x - 1, y)) {
                    xs.insert(x);
                }
                //   P
                //   x
                if !self.same_color((x, y), (x, y - 1)) {
                    ys.insert(y);
                }
            }
        }
        (xs.into_iter().collect(), ys.into_iter().collect())
    }
}

impl Index<Point> for Painting {
    type Output = Rgba;

    fn index(&self, (x, y): Point) -> &Self::Output {
        &self.pixels[(y * self.width + x) as usize]
    }
}

impl IndexMut<Point> for Painting {
    fn index_mut(&mut self, (x, y): Point) -> &mut Self::Output {
        &mut self.pixels[(y * self.width + x) as usize]
    }
}

// Full Division. Initial Config
#[derive(Serialize, Deserialize, Debug)]
pub struct InitialConfig {
    pub width: Coord,
    pub height: Coord,
    pub blocks: Vec<InitialConfigBlock>,
}

impl InitialConfig {
    pub fn new(id: ProblemId) -> Result<InitialConfig> {
        Ok(serde_json::from_str(&read_from_task_dir(format!(
            "problem/{id}.initial.json"
        ))?)?)
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct InitialConfigBlock {
    #[serde(rename = "blockId")]
    pub block_id: String,
    #[serde(rename = "bottomLeft")]
    pub bottoml_left: Vec<Coord>,
    #[serde(rename = "topRight")]
    pub top_right: Vec<Coord>,
    // TODO: Support png for v2
    pub color: Option<Rgba>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Eq, PartialEq, Hash)]
pub struct InitialBlock {
    pub id: BlockId,
    pub shape: Shape,
    pub color: Rgba,
}

#[cfg(tests)]
mod tests {

    use super::*;

    #[test]
    fn initial_config() -> Result<()> {
        let problem = Problem::new(26)?;
        assert_eq!(problem.initial_blocks.len(), 100);
        Ok(())
    }

    #[test]
    fn similarity() -> Result<()> {
        let problem = Problem::new(1)?;
        assert_eq!(problem.painting.initial_similarity(), 194_616);
        Ok(())
    }
}
