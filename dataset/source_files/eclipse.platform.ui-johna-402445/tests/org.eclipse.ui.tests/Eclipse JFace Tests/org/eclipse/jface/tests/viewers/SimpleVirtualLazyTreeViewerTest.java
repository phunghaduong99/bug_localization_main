/*******************************************************************************
 * Copyright (c) 2005, 2011 IBM Corporation and others.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     IBM Corporation - initial API and implementation
 *******************************************************************************/
package org.eclipse.jface.tests.viewers;

import org.eclipse.jface.viewers.ILazyTreeContentProvider;
import org.eclipse.jface.viewers.IStructuredSelection;
import org.eclipse.jface.viewers.StructuredSelection;
import org.eclipse.jface.viewers.StructuredViewer;
import org.eclipse.jface.viewers.TreeViewer;
import org.eclipse.jface.viewers.Viewer;
import org.eclipse.jface.viewers.ViewerComparator;
import org.eclipse.jface.viewers.ViewerSorter;
import org.eclipse.swt.SWT;
import org.eclipse.swt.widgets.Composite;
import org.eclipse.swt.widgets.Event;
import org.eclipse.swt.widgets.Listener;
import org.eclipse.swt.widgets.Tree;
import org.eclipse.swt.widgets.TreeItem;

/**
 * Tests TreeViewer's VIRTUAL support with a lazy content provider.
 *
 * @since 3.2
 */
public class SimpleVirtualLazyTreeViewerTest extends ViewerTestCase {
	private static final int NUM_ROOTS = 100;
	private static final int NUM_CHILDREN = 10;

	private boolean callbacksEnabled = true;
	private boolean printCallbacks = false;
	private int offset = 0;

	private int updateElementCallCount = 0;

	private TreeViewer<String,String> treeViewer;

	private class LazyTreeContentProvider implements ILazyTreeContentProvider<String,String> {
		/**
		 *
		 */
		private String input;

		public void updateElement(String parent, int index) {
			updateElementCallCount++;
			String parentString = parent;
			String childElement = parentString + "-" + (index+offset);
			if (printCallbacks)
				System.out.println("updateElement called for " + parent + " at " + index);
			if (callbacksEnabled) {
				getTreeViewer().replace(parent, index, childElement);
				getTreeViewer().setChildCount(childElement, NUM_CHILDREN);
			}
		}

		public void dispose() {
			// do nothing
		}

		public void inputChanged(Viewer<? extends String> viewer, String oldInput, String newInput) {
			this.input = newInput;
		}

		public String getParent(String element) {
			return null;
		}

		/* (non-Javadoc)
		 * @see org.eclipse.jface.viewers.ILazyTreeContentProvider#updateChildCount(java.lang.Object, int)
		 */
		public void updateChildCount(String element, int currentChildCount) {
			if (printCallbacks)
				System.out.println("updateChildCount called for " + element + " with " + currentChildCount);
			if (callbacksEnabled) {
				getTreeViewer().setChildCount(element, element==input?NUM_ROOTS:NUM_CHILDREN);
			}
		}
	}

	public SimpleVirtualLazyTreeViewerTest(String name) {
		super(name);
	}

	public TreeViewer<String,String> getTreeViewer() {
		return treeViewer;
	}

	/**
	 * Checks if the virtual tree / table functionality can be tested in the current settings.
	 * The virtual trees and tables rely on SWT.SetData event which is only sent if OS requests
	 * information about the tree / table. If the window is not visible (obscured by another window,
	 * outside of visible area, or OS determined that it can skip drawing), then OS request won't
	 * be send, causing automated tests to fail.
	 * See https://bugs.eclipse.org/bugs/show_bug.cgi?id=118919 .
	 */
	protected boolean setDataCalled = false;

	public void setUp() {
		super.setUp();
		processEvents(); // run events for SetData precondition test
	}

	protected void setInput() {
		String letterR = "R";
		getTreeViewer().setInput(letterR);
	}

	protected StructuredViewer createViewer(Composite parent) {
		Tree tree = new Tree(fShell, SWT.VIRTUAL | SWT.MULTI);
		treeViewer = new TreeViewer<String,String>(tree);
		treeViewer.setContentProvider(new LazyTreeContentProvider());
		tree.addListener(SWT.SetData, new Listener() {
			public void handleEvent(Event event) {
				setDataCalled = true;
			}
		});
		return treeViewer;
	}

	public void testCreation() {
		if (disableTestsBug347491)
			return;
		if (!setDataCalled) {
			System.err.println("SWT.SetData is not received. Cancelled test " + getName());
			return;
		}
		processEvents();
		assertTrue("tree should have items", getTreeViewer().getTree()
				.getItemCount() > 0);
		assertTrue("call to updateElement expected", updateElementCallCount > 0);
		assertTrue(
				"expected calls to updateElement for less than half of the items",
				updateElementCallCount < NUM_ROOTS / 2);
		assertEquals("R-0", getTreeViewer().getTree().getItem(0).getText());
	}

	public void testExpand() {
		processEvents();
		Tree tree = getTreeViewer().getTree();
		getTreeViewer().expandToLevel("R-0", 1);
		// redraw the tree - this will trigger the SetData event
		processEvents();
		assertEquals(NUM_CHILDREN, tree.getItem(0).getItemCount());
		TreeItem treeItem = tree.getItem(0).getItem(3);
		expandAndNotify(treeItem);
		// force redrawing the tree - this will trigger the SetData event
		tree.update();
		assertEquals(NUM_CHILDREN, treeItem.getItemCount());
		assertEquals(NUM_CHILDREN, treeItem.getItems().length);
		// interact();
	}

	private void expandAndNotify(TreeItem treeItem) {
		// callbacksEnabled = false;
		Tree tree = treeItem.getParent();
		tree.setRedraw(false);
		treeItem.setExpanded(true);
		try {
			Event event = new Event();
			event.item = treeItem;
			event.type = SWT.Expand;
			tree.notifyListeners(SWT.Expand, event);
		} finally {
			// callbacksEnabled = true;
			tree.setRedraw(true);
		}
	}

	public void testSetSorterOnNullInput() {
		treeViewer.setInput(null);
		treeViewer.setSorter(new ViewerSorter());
	}

	public void testSetComparatorOnNullInput(){
		treeViewer.setInput(null);
		treeViewer.setComparator(new ViewerComparator<String,String>());
	}

	/* test TreeViewer.remove(parent, index) */
	public void testRemoveAt() {
		if (disableTestsBug347491)
			return;
		if (!setDataCalled) {
			System.err.println("SWT.SetData is not received. Cancelled test " + getName());
			return;
		}
		// correct what the content provider is answering with
		treeViewer.getTree().update();
		offset = 1;
		treeViewer.remove(treeViewer.getInput(), 3);
		assertEquals(NUM_ROOTS - 1, treeViewer.getTree().getItemCount());
		treeViewer.setSelection(new StructuredSelection(new Object[] { "R-0",
				"R-1" }));
		assertEquals(2, ((IStructuredSelection) treeViewer.getSelection())
				.size());
		processEvents();
		assertTrue("expected less than " + (NUM_ROOTS / 2) + " but got "
				+ updateElementCallCount,
				updateElementCallCount < NUM_ROOTS / 2);
		updateElementCallCount = 0;
//		printCallbacks = true;
		// correct what the content provider is answering with
		offset = 2;
		treeViewer.remove(treeViewer.getInput(), 1);
		assertEquals(NUM_ROOTS - 2, treeViewer.getTree().getItemCount());
		processEvents();
		assertEquals(1, ((IStructuredSelection) treeViewer.getSelection())
				.size());
		assertEquals(1, updateElementCallCount);
//		printCallbacks = false;
	}
}
