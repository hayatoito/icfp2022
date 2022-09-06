use crate::prelude::*;

use crate::ils::*;
use crate::submission::*;

// Blocks
#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub enum Block {
    NoPaint {
        shape: Shape,
    },
    Paint {
        shape: Shape,
        color: Rgba,
        block: Box<Block>,
    },
    Cut {
        shape: Shape,
        color: Rgba,
        cut_type: CutType,
        child_blocks: Vec<Block>,
    },
}

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub enum CutType {
    P,
    LX,
    LY,
}

impl Block {
    fn lower_left(&self) -> Point {
        let shape = self.shape();
        (shape.x, shape.y)
    }

    fn moves(&self, block_id: BlockId) -> Vec<Move> {
        BlockMoves::new().moves(block_id, self)
    }

    fn shape(&self) -> &Shape {
        match self {
            Block::NoPaint { shape } => shape,
            Block::Paint { shape, .. } => shape,
            Block::Cut { shape, .. } => shape,
        }
    }

    fn paint(&self, painting: &mut Painting) {
        match self {
            Block::NoPaint { .. } => {}
            Block::Paint {
                shape,
                color,
                block,
            } => {
                for y in shape.range_y() {
                    for x in shape.range_x() {
                        painting[(x, y)] = *color;
                    }
                }
                block.paint(painting);
            }
            Block::Cut {
                shape,
                color,
                cut_type: _,
                child_blocks,
            } => {
                for y in shape.range_y() {
                    for x in shape.range_x() {
                        painting[(x, y)] = *color;
                    }
                }
                for block in child_blocks {
                    block.paint(painting);
                }
            }
        }
    }
}

struct BlockMoves {
    // We don't use merge.
    // global_counter: BlockId,
    moves: Vec<Move>,
}

impl BlockMoves {
    fn new() -> BlockMoves {
        BlockMoves {
            // global_counter: 0,
            moves: vec![],
        }
    }

    fn moves(mut self, block_id: BlockId, block: &Block) -> Vec<Move> {
        let block_ids = vec![block_id];
        self.visit(block, &block_ids);
        self.moves
    }

    fn visit(&mut self, block: &Block, block_ids: &Vec<BlockId>) {
        match block {
            Block::NoPaint { .. } => {}
            Block::Paint {
                shape: _,
                color,
                block,
            } => {
                self.moves.push(Move::Color {
                    block: block_ids.clone(),
                    color: *color,
                });
                self.visit(block, block_ids)
            }
            Block::Cut {
                shape: _,
                color: _,
                cut_type: cut,
                child_blocks,
            } => {
                self.moves.push(match cut {
                    CutType::P => Move::PCut {
                        block: block_ids.clone(),
                        // PCut point is fixed, as of now.
                        point: child_blocks[2].lower_left(),
                    },
                    CutType::LX => Move::LCut {
                        block: block_ids.clone(),
                        orientation: Orientation::X,
                        line_number: child_blocks[1].lower_left().0,
                    },
                    CutType::LY => Move::LCut {
                        block: block_ids.clone(),
                        orientation: Orientation::Y,
                        line_number: child_blocks[1].lower_left().1,
                    },
                });
                for (i, block) in child_blocks.iter().enumerate() {
                    let mut new_block_ids = block_ids.clone();
                    new_block_ids.push(i as BlockId);
                    self.visit(block, &new_block_ids);
                }
            }
        }
    }
}

// Iterative Deepning
#[derive(Debug, Hash, Eq, PartialEq)]
pub struct Node {
    shape: Shape,
    color: Rgba,
    use_color: bool,
}

#[derive(Debug, Hash, Clone)]
struct DeepSearchResult {
    block: Block,
    cost: i64,
}

type Cut = Vec<Shape>;

pub struct Solver {
    problem_id: ProblemId,
    problem: Problem,
    use_interesting_points: bool,
    try_every_color: bool,
    interesting_points: Vec<Point>,
    interesting_lcutx: Vec<Coord>,
    interesting_lcuty: Vec<Coord>,
    colors: Vec<Rgba>,
    visit_cnt: u64,
    cache: HashMap<Node, DeepSearchResult>,
    used_colors_cache: HashMap<Shape, Vec<Rgba>>,
    most_used_color_cache: HashMap<Shape, Rgba>,
    similarity_table: HashMap<Rgba, Vec<f64>>,
}

impl Solver {
    fn new(problem_id: ProblemId) -> Result<Self> {
        let problem = Problem::new(problem_id)?;

        let id = problem_id;
        // let use_interesting_points = matches!(id, 1 | 2 | 3 | 4 | 5 | 6 | 8 | 9 | 21 | 26 | 27);
        let use_interesting_points = matches!(id, 1 | 2 | 3 | 4 | 5 | 6 | 8 | 9 | 21 | 26 | 27);
        // let use_interesting_points = false;
        // let try_every_color = matches!(id, 1 | 2 | 3);
        let try_every_color = false;

        let (interesting_points, (interesting_lcutx, interesting_lcuty)) = if use_interesting_points
        {
            (
                problem.painting.interesting_pcut_point(),
                problem.painting.interesting_lcut_lines(),
            )
        } else {
            (vec![], (vec![], vec![]))
        };
        info!("interesting_points: {}", interesting_points.len());
        info!("interesting_lcutx: {}", interesting_lcutx.len());
        info!("interesting_lcuty: {}", interesting_lcuty.len());

        let mut colors = HashSet::new();
        colors.extend(problem.painting.used_colors(&problem.painting.shape()));
        colors.extend(problem.initial_blocks.iter().map(|b| b.color));

        let similarity_table = problem.painting.build_similarity_table(&colors);

        Ok(Self {
            problem_id,
            problem,
            use_interesting_points,
            try_every_color,
            interesting_points,
            interesting_lcutx,
            interesting_lcuty,
            colors: colors.into_iter().collect(),
            visit_cnt: 0,
            cache: HashMap::default(),
            used_colors_cache: HashMap::default(),
            most_used_color_cache: HashMap::default(),
            similarity_table,
        })
    }

    fn merge_initial_blocks(&mut self) {
        self.problem = self.problem.merge_initial_blocks();
    }

    fn initial_canvas(&self) -> Painting {
        let mut painting = Painting {
            width: self.problem.painting.width,
            height: self.problem.painting.height,
            pixels: vec![WHITE; self.problem.painting.size() as usize],
        };
        for initial_block in &self.problem.initial_blocks {
            for y in initial_block.shape.range_y() {
                for x in initial_block.shape.range_x() {
                    painting[(x, y)] = initial_block.color;
                }
            }
        }
        painting
    }

    #[allow(dead_code)]
    fn save_initial_canvas(&self) -> Result<()> {
        let painting = self.initial_canvas();
        painting.save_as(&format!("{}-initial.png", self.problem_id))
    }

    #[allow(dead_code)]
    fn most_used_color(&mut self, shape: &Shape) -> Option<Rgba> {
        if let Some(color) = self.most_used_color_cache.get(shape) {
            return Some(*color);
        }
        if let Some(color) = self.problem.painting.most_used_color(shape) {
            self.most_used_color_cache.insert(*shape, color);
            Some(color)
        } else {
            None
        }
    }

    #[allow(dead_code)]
    fn used_colors(&mut self, shape: &Shape) -> &[Rgba] {
        use std::collections::hash_map::Entry;

        match self.used_colors_cache.entry(*shape) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let colors = self.problem.painting.used_colors(shape);
                entry.insert(colors)
            }
        }
    }

    fn similarity(&mut self, color: &Rgba, shape: &Shape) -> Cost {
        // table is (W + 1) * (H + 1)
        let convert = |x, y| (y * (self.problem.painting.width + 1) + x) as usize;
        if let Some(table) = self.similarity_table.get(color) {
            let p0 = convert(shape.x, shape.y);
            let p1 = convert(shape.x + shape.width, shape.y);
            let p2 = convert(shape.x + shape.width, shape.y + shape.height);
            let p3 = convert(shape.x, shape.y + shape.height);
            let sum = table[p2] - table[p1] - table[p3] + table[p0];
            let cost = (sum * 0.005).round() as Cost;
            assert!(cost >= 0);
            cost
        } else {
            unreachable!()
        }
    }

    fn similarity_immutable(&self, key: (Shape, Rgba)) -> Cost {
        self.problem
            .painting
            .similarity_in_shape_with_color(&key.0, &key.1)
    }

    #[allow(dead_code)]
    fn find_swappable(&self) -> Option<(BlockId, BlockId, Cost)> {
        let mut best = 0;
        let mut best_swap = None;

        for block1 in &self.problem.initial_blocks {
            for block2 in &self.problem.initial_blocks {
                if block1.id == block2.id {
                    continue;
                }
                if block1.shape.width == block2.shape.width
                    && block1.shape.height == block2.shape.height
                {
                    let swap_cost = (3.0 * self.problem.painting.size() as f64
                        / block1.shape.size() as f64)
                        .round() as Cost;
                    let before = self.similarity_immutable((block1.shape, block1.color))
                        + self.similarity_immutable((block2.shape, block2.color));
                    let after = self.similarity_immutable((block1.shape, block2.color))
                        + self.similarity_immutable((block2.shape, block1.color));
                    let cost_diff = after + swap_cost - before;
                    if cost_diff < best {
                        best = cost_diff;
                        best_swap = Some((block1.id, block2.id, swap_cost));
                    }
                }
            }
        }
        best_swap
    }

    #[allow(dead_code)]
    fn swap_initial_blocks(&mut self) {
        // Swap initial blocks

        let mut moves = vec![];
        let mut cost = 0;
        while let Some((block1, block2, swap_cost)) = self.find_swappable() {
            moves.push(Move::Swap {
                block1: vec![block1],
                block2: vec![block2],
            });
            cost += swap_cost;
            let color1 = self.problem.initial_blocks[block1 as usize].color;
            let color2 = self.problem.initial_blocks[block2 as usize].color;
            self.problem.initial_blocks[block1 as usize].color = color2;
            self.problem.initial_blocks[block2 as usize].color = color1;
        }

        info!(
            "id: {}, Swap cnt: {}, swap cost: {}",
            self.problem_id,
            moves.len(),
            cost
        );

        self.problem.preprocess_moves.extend(moves);
        self.problem.preprocess_moves_cost += cost;
    }

    fn solve(&mut self, depth: u32) -> Solved {
        self.cache.clear();

        let initial_cost = self.problem.painting.similarity(&self.initial_canvas())
            + self.problem.preprocess_moves_cost;

        let mut solved_blocks = vec![];
        let mut total_cost = 0;

        for block in self.problem.initial_blocks.clone() {
            let solved_block = self.solve_block(&block, depth);
            total_cost += solved_block.cost;
            solved_blocks.push(solved_block);
        }

        let score = initial_cost + total_cost;

        info!(
            "id: {}, depth: {}, score: {}, visit: {}, cache {} / {}",
            self.problem_id,
            depth,
            score,
            self.visit_cnt,
            self.cache.len(),
            self.used_colors_cache.len(),
        );

        Solved {
            problem_id: self.problem_id,
            // initial_cost,
            initial_shape: self.problem.painting.shape(),
            initial_blocks: self.problem.initial_blocks.clone(),
            merge_moves: self.problem.preprocess_moves.clone(),
            score,
            solved_blocks,
        }
    }

    fn solve_block(&mut self, initial_block: &InitialBlock, depth: u32) -> SolvedBlock {
        let start = Node {
            shape: initial_block.shape,
            color: initial_block.color,
            use_color: false,
        };
        let result = self.solve_deep(start, depth);

        info!(
            "id: {}, depth: {}, Result: block_id: {}, cost: {}",
            self.problem_id, depth, initial_block.id, result.cost,
        );

        SolvedBlock {
            block: result.block,
            block_id: initial_block.id,
            cost: result.cost,
        }
    }

    fn solve_deep(&mut self, node: Node, depth: u32) -> DeepSearchResult {
        self.visit_cnt += 1;

        let no_paint = DeepSearchResult {
            block: Block::NoPaint { shape: node.shape },
            cost: 0,
        };

        if depth == 0 {
            return no_paint;
        }

        if node.shape.size() < 40 {
            return no_paint;
        }

        if let Some(res) = self.cache.get(&node) {
            return res.clone();
        }

        let similarity = self.similarity(&node.color, &node.shape);

        // info!(
        //     "similarity: {}, color: {:?}, shape: {:?}",
        //     similarity, node.color, node.shape
        // );

        if similarity == 0 {
            self.cache.insert(node, no_paint.clone());
            return no_paint;
        }

        let mut best_cost = 0;
        let mut best_block = Block::NoPaint { shape: node.shape };

        if !node.use_color {
            for color in self.color_candidates(&node.shape) {
                if color == node.color {
                    continue;
                }
                // Try to paint.
                let next_move_cost = {
                    let color_move_cost = ((COLOR_COST as f64)
                        * (self.problem.painting.size() as f64)
                        / (node.shape.size() as f64))
                        .round() as Cost;
                    let similarity_diff_cost = self.similarity(&color, &node.shape) - similarity;
                    color_move_cost + similarity_diff_cost
                };
                let next_node = Node {
                    shape: node.shape,
                    color,
                    use_color: true,
                };
                let next_result = self.solve_deep(next_node, depth - 1);
                let next_cost = next_move_cost + next_result.cost;
                if best_cost > next_cost {
                    best_cost = next_cost;
                    best_block = Block::Paint {
                        shape: node.shape,
                        color,
                        block: Box::new(next_result.block),
                    }
                }
            }
        }

        // Try PCut and LCut
        let block_size_cost = self.problem.painting.size() as f64 / node.shape.size() as f64;
        let pcut_cost =
            (self.problem.spec_version.point_cut_cost() as f64 * block_size_cost).round() as Cost;
        let lcut_cost =
            (self.problem.spec_version.line_cut_cost() as f64 * block_size_cost).round() as Cost;

        for (cut, cut_type) in self.cut_candidates(&node.shape) {
            let mut total_cost = match cut_type {
                CutType::P => pcut_cost,
                CutType::LX => lcut_cost,
                CutType::LY => lcut_cost,
            };

            let mut child_blocks = vec![];

            for sub_shape in cut {
                let next_node = Node {
                    shape: sub_shape,
                    color: node.color,
                    use_color: false,
                };
                let next_result = self.solve_deep(next_node, depth - 1);
                total_cost += next_result.cost;
                child_blocks.push(next_result.block);
            }

            if best_cost > total_cost {
                best_cost = total_cost;
                best_block = Block::Cut {
                    shape: node.shape,
                    color: node.color,
                    cut_type,
                    child_blocks,
                }
            }
        }

        let res = DeepSearchResult {
            block: best_block,
            cost: best_cost,
        };
        self.cache.insert(node, res.clone());

        res
    }

    fn color_candidates(&mut self, shape: &Shape) -> Box<dyn Iterator<Item = Rgba>> {
        if self.try_every_color {
            // Box::new(self.used_colors(shape).to_owned().into_iter())
            Box::new(self.colors.clone().into_iter())
        } else {
            Box::new(self.most_used_color(shape).into_iter())
        }
    }

    fn cut_candidates(&self, shape: &Shape) -> Box<dyn Iterator<Item = (Cut, CutType)>> {
        if self.use_interesting_points {
            let pcut = shape
                .interesting_pcut(&self.interesting_points)
                .into_iter()
                .zip(std::iter::repeat(CutType::P));

            let lcut_x = shape
                .interesting_lcutx(&self.interesting_lcutx)
                .into_iter()
                .zip(std::iter::repeat(CutType::LX));

            let lcut_y = shape
                .interesting_lcuty(&self.interesting_lcuty)
                .into_iter()
                .zip(std::iter::repeat(CutType::LY));

            Box::new(pcut.chain(lcut_x).chain(lcut_y))
        } else {
            // let pcut = shape
            //     .all_pcut()
            //     .into_iter()
            //     .zip(std::iter::repeat(CutType::P));
            let lcut_x = shape
                .all_lcut_x()
                .into_iter()
                .zip(std::iter::repeat(CutType::LX));
            let lcut_y = shape
                .all_lcut_y()
                .into_iter()
                .zip(std::iter::repeat(CutType::LY));

            // Box::new(pcut.chain(lcut_x).chain(lcut_y))
            // Box::new(pcut)
            Box::new(lcut_x.chain(lcut_y))
        }
    }
}

pub struct Solved {
    problem_id: ProblemId,
    initial_shape: Shape,
    initial_blocks: Vec<InitialBlock>,
    score: Cost,
    merge_moves: Vec<Move>,
    solved_blocks: Vec<SolvedBlock>,
}

impl Solved {
    fn save_solution(&self) -> Result<()> {
        self.save_solution_to(&format!("solution/{}-{}.txt", self.problem_id, self.score))
    }

    fn save_solution_to(&self, name: &str) -> Result<()> {
        let mut moves = self.merge_moves.clone();
        for solved_block in &self.solved_blocks {
            moves.extend(solved_block.block.moves(solved_block.block_id));
        }
        let program = Program { moves };
        debug!("program.moves.len(): {}", program.moves.len());
        write_to_task_dir(name, &program.to_string())
    }

    fn save_paint(&self) -> Result<()> {
        let mut painting = Painting {
            width: self.initial_shape.width,
            height: self.initial_shape.height,
            pixels: vec![WHITE; self.initial_shape.size() as usize],
        };
        for initial_block in &self.initial_blocks {
            for y in initial_block.shape.range_y() {
                for x in initial_block.shape.range_x() {
                    painting[(x, y)] = initial_block.color;
                }
            }
        }

        for solved_block in &self.solved_blocks {
            solved_block.block.paint(&mut painting);
        }

        painting.save_as(&format!("{}.png", self.problem_id))
    }

    pub fn save_best_if(&self) -> Result<()> {
        let results = Results::new()?;
        let is_best = match results.best(self.problem_id) {
            Some(min_cost) => {
                if self.score < min_cost {
                    println!(
                        "best: problem_id: {}, score: {} < {}",
                        self.problem_id, self.score, min_cost
                    );
                    true
                } else {
                    info!(
                        "best: problem_id: {}, score: {} >= best: {}",
                        self.problem_id, self.score, min_cost
                    );
                    false
                }
            }
            None => {
                println!(
                    "best: problem_id: {}, score: {}",
                    self.problem_id, self.score
                );
                true
            }
        };
        if is_best {
            self.save_solution_to(&format!("best/{}.txt", self.problem_id))?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct SolvedBlock {
    cost: Cost,
    block_id: BlockId,
    block: Block,
}

pub fn dump_problem(problem_id: ProblemId) -> Result<()> {
    let _problem = Problem::new(problem_id)?;
    Ok(())
}

pub fn solve(problem_id: ProblemId) -> Result<()> {
    let mut solver = Solver::new(problem_id)?;
    match solver.problem.spec_version {
        SpecVersion::V0 => {}
        SpecVersion::V1 => {
            info!("id: {problem_id}, Merge blocks...");
            solver.merge_initial_blocks();
        }
        SpecVersion::V2 => {}
    }
    // for depth in [2, 3, 4, 8, 16, 24, 32, 64, 128] {
    for depth in [2, 4, 8, 16, 32, 64] {
        info!(
            "id: {problem_id} (spec: {}), depth: {depth}, Solving...",
            solver.problem.spec_version
        );
        let solved = solver.solve(depth);

        solved.save_paint()?;
        solved.save_solution()?;
        solved.save_best_if()?;
    }
    Ok(())
}
