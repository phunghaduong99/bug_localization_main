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

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

import org.eclipse.jface.viewers.ArrayContentProvider;
import org.eclipse.jface.viewers.CellEditor;
import org.eclipse.jface.viewers.ColumnViewer;
import org.eclipse.jface.viewers.ICellModifier;
import org.eclipse.jface.viewers.StructuredViewer;
import org.eclipse.jface.viewers.TableViewer;
import org.eclipse.jface.viewers.TextCellEditor;
import org.eclipse.swt.SWT;
import org.eclipse.swt.widgets.Composite;
import org.eclipse.swt.widgets.TableColumn;

/**
 * @since 3.3
 *
 */
public class Bug180504TableViewerTest extends ViewerTestCase {

	private TableViewer<String,List<String>> tableViewer;
	/**
	 * @param name
	 */
	public Bug180504TableViewerTest(String name) {
		super(name);
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
				tableViewer.getControl().dispose();
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
		List<String> ar = new ArrayList<String>(100);
//		String[] ar = new String[100];
		for( int i = 0; i < 100; i++ ) {
			ar.add(i, i + "");
		}
		getTableViewer().setInput(ar);
	}

	private TableViewer<String,List<String>> getTableViewer() {
		return tableViewer;
	}

	public void testBug180504ApplyEditor() {
		getTableViewer().editElement((String)getTableViewer().getElementAt(0), 0);
		Method m;
		try {
			m = ColumnViewer.class.getDeclaredMethod("applyEditorValue", new Class[0]);
			m.setAccessible(true);
			m.invoke(getTableViewer(), new Object[0]);
		} catch (SecurityException e) {
			e.printStackTrace();
			fail(e.getMessage());
		} catch (NoSuchMethodException e) {
			e.printStackTrace();
			fail(e.getMessage());
		} catch (IllegalArgumentException e) {
			e.printStackTrace();
			fail(e.getMessage());
		} catch (IllegalAccessException e) {
			e.printStackTrace();
			fail(e.getMessage());
		} catch (InvocationTargetException e) {
			e.printStackTrace();
			fail(e.getMessage());

		}
	}

	public void testBug180504CancleEditor() {
		getTableViewer().editElement((String)getTableViewer().getElementAt(0), 0);
		getTableViewer().cancelEditing();
	}
}
