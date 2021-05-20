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
import java.util.List;

import org.eclipse.jface.viewers.ArrayContentProvider;
import org.eclipse.jface.viewers.CellEditor;
import org.eclipse.jface.viewers.ColumnViewer;
import org.eclipse.jface.viewers.ICellModifier;
import org.eclipse.jface.viewers.StructuredViewer;
import org.eclipse.jface.viewers.TableViewer;
import org.eclipse.jface.viewers.TextCellEditor;
import org.eclipse.jface.viewers.ViewerCell;
import org.eclipse.swt.SWT;
import org.eclipse.swt.widgets.Composite;
import org.eclipse.swt.widgets.TableColumn;

/**
 * @since 3.3
 *
 */
public class Bug203657TableViewerTest extends ViewerTestCase {

	private TableViewer<String,List<String>> tableViewer;

	/**
	 * @param name
	 */
	public Bug203657TableViewerTest(String name) {
		super(name);
		// TODO Auto-generated constructor stub
	}

	protected StructuredViewer createViewer(Composite parent) {
		tableViewer = new TableViewer<String,List<String>>(parent, SWT.FULL_SELECTION);
		tableViewer.setContentProvider(new ArrayContentProvider<String>(String.class));
		tableViewer.setCellEditors(new CellEditor[] { new TextCellEditor(
				tableViewer.getTable()) });
		tableViewer.setColumnProperties(new String[] { "0" });
		tableViewer.setCellModifier(new ICellModifier() {
			public boolean canModify(Object element, String property) {
				return true;
			}

			public Object getValue(Object element, String property) {
				return "";
			}

			public void modify(Object element, String property, Object value) {
			}

		});

	    new TableColumn(tableViewer.getTable(), SWT.NONE).setWidth(200);

		return tableViewer;
	}

	protected void setUpModel() {
		// don't do anything here - we are not using the normal fModel and
		// fRootElement
	}

	protected void setInput() {

		ArrayList<String> ar = new ArrayList<String>(100);
		for( int i = 0; i < 100; i++ ) {
			ar.add(i, i + "");
		}
		getTableViewer().setInput(ar);
	}

	private TableViewer<String,List<String>> getTableViewer() {
		return tableViewer;
	}

	public void testBug203657() {
		try {
			Field f = ColumnViewer.class.getDeclaredField("cell");
			f.setAccessible(true);
			ViewerCell<String> cell = (ViewerCell<String>) f.get(getTableViewer());
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
