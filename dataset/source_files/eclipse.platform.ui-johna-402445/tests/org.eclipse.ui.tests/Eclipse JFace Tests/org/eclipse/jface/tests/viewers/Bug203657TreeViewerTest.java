/*******************************************************************************
 * Copyright (c) 2007 IBM Corporation and others.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     IBM Corporation - initial API and implementation
 ******************************************************************************/

package org.eclipse.jface.tests.viewers;

import java.lang.reflect.Field;
import java.util.ArrayList;

import org.eclipse.jface.viewers.CellEditor;
import org.eclipse.jface.viewers.ColumnViewer;
import org.eclipse.jface.viewers.ICellModifier;
import org.eclipse.jface.viewers.ITreeContentProvider;
import org.eclipse.jface.viewers.StructuredViewer;
import org.eclipse.jface.viewers.TextCellEditor;
import org.eclipse.jface.viewers.TreeViewer;
import org.eclipse.jface.viewers.Viewer;
import org.eclipse.jface.viewers.ViewerCell;
import org.eclipse.swt.SWT;
import org.eclipse.swt.widgets.Composite;
import org.eclipse.swt.widgets.TreeColumn;

/**
 * @since 3.3
 *
 */
public class Bug203657TreeViewerTest extends ViewerTestCase {

	private TreeViewer<MyModel,MyModel> treeViewer;

	public class MyModel {
		public MyModel parent;

		public ArrayList<MyModel> child = new ArrayList<MyModel>();

		public int counter;

		public MyModel(int counter, MyModel parent) {
			this.parent = parent;
			this.counter = counter;
		}

		public String toString() {
			String rv = "Item ";
			if (parent != null) {
				rv = parent.toString() + ".";
			}

			rv += counter;

			return rv;
		}
	}
	/**
	 * @param name
	 */
	public Bug203657TreeViewerTest(String name) {
		super(name);
		// TODO Auto-generated constructor stub
	}

	protected StructuredViewer createViewer(Composite parent) {
		treeViewer = new TreeViewer<MyModel,MyModel>(parent, SWT.FULL_SELECTION);

		treeViewer.setContentProvider(new ITreeContentProvider<MyModel,MyModel>() {

			public MyModel[] getElements(MyModel inputElement) {
				MyModel[] children =  new MyModel[inputElement.child.size()];
				return inputElement.child.toArray(children);
			}

			public void dispose() {

			}

			public void inputChanged(Viewer<? extends MyModel> viewer, MyModel oldInput,
					MyModel newInput) {

			}

			public MyModel[] getChildren(MyModel parentElement) {
				return getElements(parentElement);
			}

			public MyModel getParent(MyModel element) {
				if (element == null) {
					return null;
				}

				return element.parent;
			}

			public boolean hasChildren(MyModel element) {
				return element.child.size() > 0;
			}
		});

		treeViewer.setCellEditors(new CellEditor[] { new TextCellEditor(
				treeViewer.getTree()) });
		treeViewer.setColumnProperties(new String[] { "0" });
		treeViewer.setCellModifier(new ICellModifier() {
			public boolean canModify(Object element, String property) {
				return true;
			}

			public Object getValue(Object element, String property) {
				return "";
			}

			public void modify(Object element, String property, Object value) {
			}

		});

		new TreeColumn(treeViewer.getTree(), SWT.NONE).setWidth(200);

		return treeViewer;
	}

	protected void setUpModel() {
		// don't do anything here - we are not using the normal fModel and
		// fRootElement
	}

	protected void setInput() {
		MyModel root = new MyModel(0, null);
		root.counter = 0;

		MyModel tmp;
		for (int i = 1; i < 100; i++) {
			tmp = new MyModel(i, root);
			root.child.add(tmp);
			for (int j = 1; j < i; j++) {
				tmp.child.add(new MyModel(j, tmp));
			}
		}

		getTreeViewer().setInput(root);
	}

	private TreeViewer<MyModel,MyModel> getTreeViewer() {
		return treeViewer;
	}

	public void testBug203657() {
		try {
			Field f = ColumnViewer.class.getDeclaredField("cell");
			f.setAccessible(true);
			ViewerCell<MyModel> cell = (ViewerCell<MyModel>) f.get(getTreeViewer());
			assertNull(cell.getElement());
			assertNull(cell.getViewerRow());
			assertEquals(0, cell.getColumnIndex());
		} catch (SecurityException e) {
			fail(e.getMessage());
		} catch (NoSuchFieldException e) {
			fail(e.getMessage());
		} catch (IllegalArgumentException e) {
			fail(e.getMessage());
		} catch (IllegalAccessException e) {
			fail(e.getMessage());
		}
	}


}
