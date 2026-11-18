//! # Ternary search tree implementation.
//!
//! - **Author:** Daniel Starke
//! - **Copyright:** Copyright 2026 Daniel Starke
//! - **Date:** 2026-02-16
//! - **Version:** 2026-03-10

#[derive(Clone)]
struct TstNode<T> {
	/// ASCII character value stored in this node
	value: u8,
	/// Data stored at this node if it is a terminal
	data: Option<T>,
	/// Left child (characters < value)
	left: Option<Box<TstNode<T>>>,
	/// Middle child (next character)
	middle: Option<Box<TstNode<T>>>,
	/// Right child (characters > value)
	right: Option<Box<TstNode<T>>>,
}

impl<T> TstNode<T> {
	fn new(value: u8) -> Self {
		TstNode {
			value,
			data: None,
			left: None,
			middle: None,
			right: None,
		}
	}
}

/// Returns (length, data) of longest match ASCII case-sensitive.
fn find<T: Clone>(node: Option<&TstNode<T>>, bytes: &[u8], index: usize) -> Option<(usize, T)> {
	let node = node?;
	if index >= bytes.len() {
		return None;
	}
	let b = bytes[index];
	if b < node.value {
		find(node.left.as_deref(), bytes, index)
	} else if b > node.value {
		find(node.right.as_deref(), bytes, index)
	} else {
		let longer = find(node.middle.as_deref(), bytes, index + 1);
		if longer.is_some() {
			longer
		} else {
			node.data.as_ref().map(|d| (index + 1, d.clone()))
		}
	}
}

/// Symbol table.
///
/// Complexity: O(log(n) + k) where n is number of strings, k is string length
#[derive(Clone)]
pub struct SymbolMap<T> {
	root: Option<Box<TstNode<T>>>,
}

impl<T: Clone> SymbolMap<T> {
	/// Create a new empty symbol map
	pub fn new() -> Self {
		SymbolMap { root: None }
	}

	/// Add a symbol to the map
	pub fn add(&mut self, symbol: &str, value: T) {
		let bytes = symbol.as_bytes();
		if bytes.is_empty() {
			return;
		}
		Self::add_impl(&mut self.root, bytes, 0, value);
	}

	/// Recursive add implementation
	fn add_impl(node: &mut Option<Box<TstNode<T>>>, bytes: &[u8], index: usize, value: T) {
		if index >= bytes.len() {
			return;
		}
		let b = bytes[index];
		if node.is_none() {
			*node = Some(Box::new(TstNode::new(b)));
		}
		let node_ref = node.as_mut().unwrap();
		if b < node_ref.value {
			Self::add_impl(&mut node_ref.left, bytes, index, value);
		} else if b > node_ref.value {
			Self::add_impl(&mut node_ref.right, bytes, index, value);
		} else if (index + 1) == bytes.len() {
			// Terminal - store data here, middle child is unaffected
			node_ref.data = Some(value);
		} else {
			// Non-terminal - recurse into middle child
			Self::add_impl(&mut node_ref.middle, bytes, index + 1, value);
		}
	}

	/// Returns (length, data) of longest match ASCII case-sensitive.
	#[inline]
	pub fn find(&self, bytes: &[u8]) -> Option<(usize, T)> {
		find(self.root.as_deref(), bytes, 0)
	}
}

impl<T: Clone> Default for SymbolMap<T> {
	fn default() -> Self {
		Self::new()
	}
}
